"""患者新号冲突回归测试：两个不同原 PatientID 派生出同一新号时，
- 同批、跨批、并发上传都不得共号；
- 同一原 ID 重传、失败重试、重启后不变号；
- 影像（顶层与嵌套）、数据库映射、下载清单三者一致；
- 无法形成无冲突映射时整批拒绝，已有可下载批次不受影响。

测试中通过 monkeypatch derive_patient_id 为定值来强制制造碰撞。
"""
from __future__ import annotations

import csv
import io
import re
import threading
import zipfile
from pathlib import Path

import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.tag import Tag
from werkzeug.datastructures import MultiDict

from app import identifiers

COLLIDING_ID = "P000000000042"
NEW_ID_RE = re.compile(r"^P\d{12}$")


# ---------------------------------------------------------------- 工具函数
def _make_dcm(src: Path, dst: Path, patient_id: str, sop_uid: str,
              nested_patient_id: str | None = None) -> None:
    """基于样例生成指定 PatientID / SOP UID 的测试文件。"""
    ds = pydicom.dcmread(str(src))
    ds.PatientID = patient_id
    ds.SOPInstanceUID = sop_uid
    ds.file_meta.MediaStorageSOPInstanceUID = sop_uid
    if nested_patient_id is not None:
        item = Dataset()
        item.add_new(Tag(0x0010, 0x0020), "LO", nested_patient_id)
        ds.add_new(Tag(0x0040, 0x0275), "SQ", pydicom.Sequence([item]))
    ds.save_as(str(dst), write_like_original=False)


def _upload(client, paths) -> str:
    data = MultiDict()
    handles = []
    for p in paths:
        fh = open(p, "rb")
        handles.append(fh)
        data.add("dicom_files", (fh, Path(p).name, "application/dicom"))
    resp = client.post("/upload", data=data, content_type="multipart/form-data",
                       follow_redirects=False)
    for fh in handles:
        fh.close()
    assert resp.status_code == 302
    return resp.headers["Location"].rstrip("/").split("/")[-1]


def _download(client, batch_id: str) -> zipfile.ZipFile:
    resp = client.get(f"/batches/{batch_id}/download")
    assert resp.status_code == 200
    return zipfile.ZipFile(io.BytesIO(resp.data))


def _manifest_rows(zf: zipfile.ZipFile) -> list[dict[str, str]]:
    text = zf.read("manifest.csv").decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _image_patient_ids(zf: zipfile.ZipFile) -> dict[str, str]:
    """zip 内每个 dcm 的（文件名 -> 顶层 PatientID）。"""
    out = {}
    for name in zf.namelist():
        if name.endswith(".dcm"):
            ds = pydicom.dcmread(io.BytesIO(zf.read(name)))
            out[name] = str(ds.PatientID)
    return out


def _db_patient_map() -> dict[str, str]:
    from app.db import get_db
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT original_patient_id, new_patient_id FROM patient_mappings"
        ).fetchall()
        return {r["original_patient_id"]: r["new_patient_id"] for r in rows}
    finally:
        conn.close()


@pytest.fixture()
def force_collision(monkeypatch):
    """强制所有原 PatientID 派生出同一新号（模拟哈希碰撞）。"""
    monkeypatch.setattr(
        identifiers, "derive_patient_id", lambda pid: COLLIDING_ID
    )
    return COLLIDING_ID


# ---------------------------------------------------------------- 同批冲突
def test_same_batch_collision_resolved(client, sample_files, tmp_path,
                                       force_collision):
    f1 = tmp_path / "a.dcm"
    f2 = tmp_path / "b.dcm"
    _make_dcm(sample_files / "ct_001.dcm", f1, "PAT-A", "1.2.826.0.1.9999.8800.1")
    _make_dcm(sample_files / "ct_002.dcm", f2, "PAT-B", "1.2.826.0.1.9999.8800.2")

    batch_id = _upload(client, [f1, f2])
    zf = _download(client, batch_id)
    rows = _manifest_rows(zf)
    assert len(rows) == 2

    new_ids = {r["original_patient_id"]: r["new_patient_id"] for r in rows}
    # 不同原 ID 不得共号；按排序顺序确定性占用派生槽位与下一个空位
    assert new_ids["PAT-A"] == "P000000000042"
    assert new_ids["PAT-B"] == "P000000000043"
    for nid in new_ids.values():
        assert NEW_ID_RE.match(nid), nid

    # 影像顶层 PatientID 与清单一致
    image_ids = _image_patient_ids(zf)
    for r in rows:
        assert image_ids[r["new_sop_uid"] + ".dcm"] == \
            new_ids[r["original_patient_id"]]

    # 数据库映射与清单一致
    db_map = _db_patient_map()
    assert db_map["PAT-A"] == "P000000000042"
    assert db_map["PAT-B"] == "P000000000043"


# ---------------------------------------------------------------- 跨批冲突与重传
def test_cross_batch_collision_and_reupload_stable(client, sample_files,
                                                   tmp_path, force_collision):
    f1 = tmp_path / "a.dcm"
    f2 = tmp_path / "b.dcm"
    _make_dcm(sample_files / "ct_001.dcm", f1, "PAT-A", "1.2.826.0.1.9999.8801.1")
    _make_dcm(sample_files / "ct_002.dcm", f2, "PAT-B", "1.2.826.0.1.9999.8801.2")

    b1 = _upload(client, [f1])
    rows1 = _manifest_rows(_download(client, b1))
    assert rows1[0]["new_patient_id"] == "P000000000042"

    # 第二批：不同原 ID 派生值碰撞 -> 必须让位，不得沿用第一人的号，清单不得为空
    b2 = _upload(client, [f2])
    rows2 = _manifest_rows(_download(client, b2))
    nid_b = rows2[0]["new_patient_id"]
    assert nid_b == "P000000000043"
    assert nid_b != rows1[0]["new_patient_id"]
    img2 = _image_patient_ids(_download(client, b2))
    assert list(img2.values()) == [nid_b]

    # 同一原 ID 重传：不变号
    b3 = _upload(client, [f2])
    rows3 = _manifest_rows(_download(client, b3))
    assert rows3[0]["new_patient_id"] == nid_b

    # 已有可下载批次照常提供，且第一人号码不变
    rows1_again = _manifest_rows(_download(client, b1))
    assert rows1_again[0]["new_patient_id"] == "P000000000042"


# ---------------------------------------------------------------- 并发冲突
def test_concurrent_colliding_uploads_never_share(client, sample_files,
                                                  tmp_path, force_collision):
    n = 6
    files = []
    for i in range(n):
        f = tmp_path / f"p{i}.dcm"
        _make_dcm(sample_files / "ct_001.dcm", f, f"PAT-C-{i}",
                  f"1.2.826.0.1.9999.8802.{i + 1}")
        files.append(f)

    results: list[str] = []
    errors: list[Exception] = []

    def worker(path: Path) -> None:
        try:
            results.append(_upload(client, [path]))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f,)) for f in files]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == n

    # 全部批次可下载，且新号两两不同：恰好占满 42..47 连续探测槽位
    new_ids = set()
    for bid in results:
        rows = _manifest_rows(_download(client, bid))
        assert len(rows) == 1
        nid = rows[0]["new_patient_id"]
        assert NEW_ID_RE.match(nid)
        new_ids.add(nid)
    assert new_ids == {f"P{42 + k:012d}" for k in range(n)}

    # 数据库中同样是一一映射，无共号
    db_map = _db_patient_map()
    assert len(db_map) == n
    assert set(db_map.values()) == new_ids


# ---------------------------------------------------------------- 探测耗尽整批拒绝
def test_probe_exhaustion_rejects_whole_batch(client, sample_files, tmp_path,
                                              monkeypatch, force_collision):
    monkeypatch.setattr(identifiers, "MAX_PATIENT_ID_ATTEMPTS", 2)

    made = []
    for i, pid in enumerate(("PAT-X", "PAT-Y", "PAT-Z")):
        f = tmp_path / f"{pid}.dcm"
        _make_dcm(sample_files / "ct_001.dcm", f, pid,
                  f"1.2.826.0.1.9999.8803.{i + 1}")
        made.append(f)

    # 前两位患者占满仅有的两个候选槽位
    b1 = _upload(client, [made[0]])
    b2 = _upload(client, [made[1]])
    assert _manifest_rows(_download(client, b1))[0]["new_patient_id"] == \
        "P000000000042"
    assert _manifest_rows(_download(client, b2))[0]["new_patient_id"] == \
        "P000000000043"

    # 第三位患者无空位 -> 整批拒绝，绝不发布
    b3 = _upload(client, [made[2]])
    assert client.get(f"/batches/{b3}/download").status_code == 409
    detail = client.get(f"/batches/{b3}")
    assert b"failed" in detail.data
    assert "无法形成无冲突映射".encode() in detail.data

    # 已有可下载批次照常提供
    assert client.get(f"/batches/{b1}/download").status_code == 200
    assert client.get(f"/batches/{b2}/download").status_code == 200

    # 被拒绝的患者不得留下映射（下次有空位时可重新分配）
    assert "PAT-Z" not in _db_patient_map()


# ---------------------------------------------------------------- 失败重试不变号
def test_failed_retry_keeps_patient_id(client, sample_files, tmp_path,
                                       monkeypatch, force_collision):
    from app import services

    f1 = tmp_path / "a.dcm"
    f2 = tmp_path / "b.dcm"
    _make_dcm(sample_files / "ct_001.dcm", f1, "PAT-A", "1.2.826.0.1.9999.8804.1")
    _make_dcm(sample_files / "ct_002.dcm", f2, "PAT-B", "1.2.826.0.1.9999.8804.2")

    b1 = _upload(client, [f1])
    assert _manifest_rows(_download(client, b1))[0]["new_patient_id"] == \
        "P000000000042"

    # 第二批在打包阶段失败（模拟断电）：患者预留已提交，批次不可下载
    def broken_zip(batch_id, plan):  # noqa: ANN001
        raise RuntimeError("模拟断电")

    monkeypatch.setattr(services, "_build_zip", broken_zip, raising=True)
    b2 = _upload(client, [f2])
    assert client.get(f"/batches/{b2}/download").status_code == 409
    assert _db_patient_map()["PAT-B"] == "P000000000043"

    # 撤销全部补丁（含碰撞强制）：即便派生函数已恢复，重试仍沿用已登记号码
    monkeypatch.undo()
    b3 = _upload(client, [f2])
    rows3 = _manifest_rows(_download(client, b3))
    assert rows3[0]["new_patient_id"] == "P000000000043"
    img3 = _image_patient_ids(_download(client, b3))
    assert list(img3.values()) == ["P000000000043"]


# ---------------------------------------------------------------- 重启后不变号
def test_restart_keeps_patient_ids(app, client, sample_files, tmp_path,
                                   force_collision):
    f1 = tmp_path / "a.dcm"
    f2 = tmp_path / "b.dcm"
    _make_dcm(sample_files / "ct_001.dcm", f1, "PAT-A", "1.2.826.0.1.9999.8805.1")
    _make_dcm(sample_files / "ct_002.dcm", f2, "PAT-B", "1.2.826.0.1.9999.8805.2")

    b1 = _upload(client, [f1])
    b2 = _upload(client, [f2])
    before = _db_patient_map()
    assert before == {"PAT-A": "P000000000042", "PAT-B": "P000000000043"}

    # 模拟服务重启：同一数据目录上重建应用（启动恢复不得改动已登记映射）
    from app.web import create_app
    restarted_client = create_app().test_client()

    assert _db_patient_map() == before
    # 重启后重传同一原 ID：号码不变
    b3 = _upload(restarted_client, [f2])
    rows3 = _manifest_rows(_download(restarted_client, b3))
    assert rows3[0]["new_patient_id"] == "P000000000043"
    # 重启前已可下载的批次照常提供
    assert restarted_client.get(f"/batches/{b1}/download").status_code == 200
    assert restarted_client.get(f"/batches/{b2}/download").status_code == 200


# ---------------------------------------------------------------- 嵌套 PatientID 一致
def test_nested_patient_id_consistent_everywhere(client, sample_files,
                                                 tmp_path):
    f1 = tmp_path / "top.dcm"
    f2 = tmp_path / "ref.dcm"
    # 文件 1：顶层 PAT-TOP，嵌套 PAT-SHARED；文件 2：顶层 PAT-SHARED
    _make_dcm(sample_files / "ct_001.dcm", f1, "PAT-TOP",
              "1.2.826.0.1.9999.8806.1", nested_patient_id="PAT-SHARED")
    _make_dcm(sample_files / "ct_002.dcm", f2, "PAT-SHARED",
              "1.2.826.0.1.9999.8806.2")

    batch_id = _upload(client, [f1, f2])
    zf = _download(client, batch_id)
    rows = _manifest_rows(zf)
    new_by_orig = {r["original_patient_id"]: r["new_patient_id"] for r in rows}

    # 清单：两个原 ID 都有非空新号且互不共号
    assert set(new_by_orig) == {"PAT-TOP", "PAT-SHARED"}
    assert all(NEW_ID_RE.match(v) for v in new_by_orig.values())
    assert new_by_orig["PAT-TOP"] != new_by_orig["PAT-SHARED"]

    # 影像：文件 1 的嵌套 PatientID 与文件 2 的顶层 PatientID 同号，
    # 且都等于数据库中 PAT-SHARED 的登记值
    db_map = _db_patient_map()
    for r in rows:
        ds = pydicom.dcmread(io.BytesIO(zf.read(r["new_sop_uid"] + ".dcm")))
        assert str(ds.PatientID) == db_map[r["original_patient_id"]]
        if r["original_patient_id"] == "PAT-TOP":
            nested = ds[Tag(0x0040, 0x0275)].value[0]
            nested_new = str(nested[Tag(0x0010, 0x0020)].value)
            assert nested_new == db_map["PAT-SHARED"]
            assert nested_new == new_by_orig["PAT-SHARED"]

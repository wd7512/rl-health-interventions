"""Unified data loading module for 12 open wearable/health datasets.

Each loader downloads (if needed) from its source and returns a polars DataFrame
with standardised columns (user_id, timestamp, …).  Missing credentials are
handled gracefully — the function logs a warning and returns ``None``.
"""

from __future__ import annotations

import logging
import os
import tarfile
import zipfile
from io import StringIO
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin

import numpy as np
import polars as pl
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_path(data_dir: str | os.PathLike, name: str, *parts: str) -> Path:
    """Return a path inside ``data_dir / name / *parts``, creating dirs."""
    return _ensure_dir(Path(data_dir) / name / Path(*parts).parent) / Path(*parts).name


def _download_file(
    url: str,
    dest: str | os.PathLike,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Download *url* to *dest* (overwrite if exists).

    Returns the resolved destination path.
    """
    dest = Path(dest)
    logger.info("Downloading %s -> %s", url, dest)
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return dest


def _check_kaggle_auth() -> bool:
    """Return ``True`` if a Kaggle credential file or env-var exists."""
    home = Path.home()
    if (home / ".kaggle" / "kaggle.json").exists():
        return True
    if (home / ".kaggle" / "access_token").exists():
        return True
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    return False


def _check_physionet_auth() -> bool:
    """Return ``True`` if PhysioNet credentials are configured."""
    if os.environ.get("PHYSIONET_USERNAME") and os.environ.get("PHYSIONET_PASSWORD"):
        return True
    if os.environ.get("WFDB_USERNAME") and os.environ.get("WFDB_PASSWORD"):
        return True
    return False


# ---------------------------------------------------------------------------
# Kaggle helpers via kagglehub
# ---------------------------------------------------------------------------


def _kaggle_download(
    handle: str, data_dir: str | os.PathLike, name: str
) -> Path | None:
    """Download a Kaggle dataset via kagglehub, return the actual download path.

    kagglehub caches downloads under ``~/.cache/kagglehub/`` internally, so
    the returned :class:`~pathlib.Path` points directly into that cache tree.

    Returns ``None`` if credentials are missing.
    """
    if not _check_kaggle_auth():
        logger.warning(
            "Kaggle credentials not found.  Set up ~/.kaggle/kaggle.json or "
            "the KAGGLE_API_TOKEN env var.  Skipping %s.",
            handle,
        )
        return None

    try:
        import kagglehub  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        logger.warning("kagglehub not installed; cannot download %s", handle)
        return None

    # Check kagglehub cache first — avoids SSL/API errors when data is cached
    try:
        owner, dataset = handle.split("/", 1)
        cache_root = Path.home() / ".cache" / "kagglehub" / "datasets" / owner / dataset / "versions"
        if cache_root.exists():
            versions = sorted(cache_root.iterdir())
            if versions:
                cached_path = versions[-1]
                logger.info("Found cached Kaggle dataset %s at %s", handle, cached_path)
                return cached_path
    except Exception:
        pass

    try:
        path = kagglehub.dataset_download(handle)
        if path is None:
            logger.warning("kagglehub returned None for %s", handle)
            return None
        dl_path = Path(path)
        logger.info("Kaggle dataset %s available at %s", handle, dl_path)
        return dl_path
    except Exception:
        logger.exception("Failed to download Kaggle dataset %s", handle)
        return None


# ---------------------------------------------------------------------------
# PhysioNet helper via wfdb
# ---------------------------------------------------------------------------


def _physionet_download(
    db_name: str,
    data_dir: str | os.PathLike,
    name: str,
) -> Path | None:
    """Download a PhysioNet database via wfdb.

    Some PhysioNet datasets require accepted terms of use (handled via
    ``PHYSIONET_USERNAME`` / ``PHYSIONET_PASSWORD`` env vars).
    """
    dest_dir = _ensure_dir(Path(data_dir) / name)
    maybe_marker = dest_dir / ".physionet_done"
    if maybe_marker.exists():
        logger.info("PhysioNet db %s already cached at %s", db_name, dest_dir)
        return dest_dir

    # Collect credentials from env vars — wfdb passes these to the PhysioNet API
    username = os.environ.get("PHYSIONET_USERNAME") or os.environ.get("WFDB_USERNAME")
    password = os.environ.get("PHYSIONET_PASSWORD") or os.environ.get("WFDB_PASSWORD")

    try:
        import wfdb  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        logger.warning("wfdb not installed; cannot download %s", db_name)
        return None

    try:
        wfdb.dl_database(
            db_name,
            dl_dir=str(dest_dir),
            overwrite=False,
            username=username,
            password=password,
        )
        maybe_marker.touch()
        logger.info("PhysioNet db %s downloaded to %s", db_name, dest_dir)
        return dest_dir
    except Exception:
        logger.warning(
            "Failed to download PhysioNet db %s (may need accepted terms or "
            "credentials).  Skipping.",
            db_name,
        )
        return None


# ---------------------------------------------------------------------------
# Directory-scanning helpers
# ---------------------------------------------------------------------------


def _find_csv_files(root: Path) -> list[Path]:
    """Recursively find all CSV files under *root*."""
    return sorted(root.rglob("*.csv"))


def _find_tsv_files(root: Path) -> list[Path]:
    """Recursively find all TSV files under *root*."""
    return sorted(root.rglob("*.tsv"))


# ===================================================================
# Loader: PMData (Kaggle)
# ===================================================================


def load_pmdata(
    data_dir: str = "data",
) -> pl.DataFrame | None:
    """Load PMData sports-logging dataset from Kaggle.

    Expected columns: ``user_id, timestamp, heart_rate, heart_rate_variability,
    step_count, sleep_duration, sleep_stages, recovery_score, spo2,
    body_temperature, stress_level``.
    """
    dest = _kaggle_download(
        "vlbthambawita/pmdata-a-sports-logging-dataset", data_dir, "pmdata"
    )
    if dest is None:
        return None

    # PMData is organised as: osfstorage-archive/pmdata/pXX/fitbit/sleep_score.csv
    # where XX = 01..16.  Load every fitbit/sleep_score.csv across participants.
    sleep_score_files = sorted(dest.rglob("fitbit/sleep_score.csv"))
    if not sleep_score_files:
        # fallback: any CSV files
        csv_files = _find_csv_files(dest)
        if not csv_files:
            logger.warning("No CSV files found in PMData download")
            return None
        sleep_score_files = csv_files

    frames: list[pl.DataFrame] = []
    for f in sleep_score_files:
        try:
            df = pl.read_csv(f, try_parse_dates=True)
            if "user_id" not in df.columns:
                # Extract user id from parent directory name (e.g. "p10")
                user = f.parent.parent.name  # sleep_score.csv → fitbit → p10
                df = df.with_columns(pl.lit(user).alias("user_id"))
            frames.append(df)
        except Exception:
            logger.debug("Skipping unreadable CSV %s", f)

    if not frames:
        return None

    result = pl.concat(frames, how="diagonal")
    # Standardise timestamp column
    for ts_col in ("date", "timestamp", "time", "datetime"):
        if ts_col in result.columns:
            result = result.rename({ts_col: "timestamp"})
            break
    if "timestamp" in result.columns and result["timestamp"].dtype != pl.Datetime:
        result = result.with_columns(
            result["timestamp"].cast(pl.Datetime, strict=False)
        )
    logger.info("PMData loaded: %d rows, %d columns", result.height, result.width)
    return result


# ===================================================================
# Loader: HARTH (HuggingFace)
# ===================================================================


def load_harth(
    data_dir: str = "data",
) -> pl.DataFrame | None:
    """Load HARTH dataset from HuggingFace.

    Expected columns: ``user_id, timestamp, thigh_accel_x/y/z, back_accel_x/y/z,
    activity``.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        logger.warning("datasets library not installed; cannot load HARTH")
        return None

    try:
        cache_dir = _ensure_dir(Path(data_dir) / "harth")
        ds = load_dataset(
            "josefheidler/har_adults_2021-harth", cache_dir=str(cache_dir)
        )
    except Exception:
        logger.exception("Failed to load HARTH dataset from HuggingFace")
        return None

    # Determine split — commonly "train" or the only split
    split = list(ds.keys())[0] if ds else None
    if split is None:
        return None

    raw = ds[split].to_polars()  # type: ignore[union-attr]
    # to_polars can return an Iterator for large datasets → collect
    if isinstance(raw, pl.DataFrame):
        df = raw
    else:
        df = pl.concat(list(raw), how="diagonal")

    # Standardise columns
    col_map: dict[str, str] = {}
    if "subject" in df.columns and "user_id" not in df.columns:
        col_map["subject"] = "user_id"
    for ts_col in ("time", "datetime", "date"):
        if ts_col in df.columns and "timestamp" not in df.columns:
            col_map[ts_col] = "timestamp"
    if col_map:
        df = df.rename(col_map)
    if "timestamp" in df.columns and df["timestamp"].dtype != pl.Datetime:
        df = df.with_columns(df["timestamp"].cast(pl.Datetime, strict=False))

    logger.info("HARTH loaded: %d rows, %d columns", df.height, df.width)
    return df


# ===================================================================
# Loader: BIDSleep (PhysioNet)
# ===================================================================


def load_bidsleep(
    data_dir: str = "data",
) -> pl.DataFrame | None:
    """Load BIDSleep dataset from PhysioNet (open access).

    Expected columns: ``user_id, night, timestamp, heart_rate, accel_x/y/z,
    sleep_stage, sleep_stage_onset``.
    """
    dest = _physionet_download("bidsleep-dataset/1.0.0", data_dir, "bidsleep")
    if dest is None:
        return None

    # BIDSleep uses BIDS format — look for *physio* TSV files
    tsv_files = _find_tsv_files(dest)
    csv_files = _find_csv_files(dest)
    all_files = csv_files + tsv_files
    if not all_files:
        logger.warning("No CSV/TSV files found in BIDSleep download")
        return None

    frames: list[pl.DataFrame] = []
    for f in all_files:
        try:
            sep = "," if f.suffix == ".csv" else "\t"
            df = pl.read_csv(f, separator=sep, try_parse_dates=True)
            if df.width > 1:
                frames.append(df)
        except Exception:
            logger.debug("Skipping unreadable file %s", f)

    if not frames:
        return None
    result = pl.concat(frames, how="diagonal")
    # Standardise
    for ts_col in ("date", "time", "timestamp", "datetime"):
        if ts_col in result.columns:
            result = result.rename({ts_col: "timestamp"})
            break
    if "timestamp" in result.columns and result["timestamp"].dtype != pl.Datetime:
        result = result.with_columns(
            result["timestamp"].cast(pl.Datetime, strict=False)
        )
    logger.info("BIDSleep loaded: %d rows, %d columns", result.height, result.width)
    return result


# ===================================================================
# Loader: ScientISST MOVE (PhysioNet)
# ===================================================================


def load_scientisst_move(
    data_dir: str = "data",
) -> pl.DataFrame | None:
    """Load ScientISST MOVE dataset from PhysioNet (open access).

    Expected columns: ``user_id, timestamp, accel_x/y/z, gyro_x/y/z, eda,
    heart_rate, skin_temperature, activity_label``.
    """
    dest = _physionet_download(
        "scientisst-move-biosignals/1.0.1", data_dir, "scientisst_move"
    )
    if dest is None:
        return None

    csv_files = _find_csv_files(dest)
    tsv_files = _find_tsv_files(dest)
    all_files = csv_files + tsv_files
    if not all_files:
        logger.warning("No CSV/TSV files found in ScientISST download")
        return None

    frames: list[pl.DataFrame] = []
    for f in all_files:
        try:
            sep = "," if f.suffix == ".csv" else "\t"
            df = pl.read_csv(f, separator=sep, try_parse_dates=True)
            if df.width > 1:
                frames.append(df)
        except Exception:
            logger.debug("Skipping unreadable file %s", f)

    if not frames:
        return None
    result = pl.concat(frames, how="diagonal")
    for ts_col in ("date", "time", "timestamp", "datetime"):
        if ts_col in result.columns:
            result = result.rename({ts_col: "timestamp"})
            break
    if "timestamp" in result.columns and result["timestamp"].dtype != pl.Datetime:
        result = result.with_columns(
            result["timestamp"].cast(pl.Datetime, strict=False)
        )
    logger.info(
        "ScientISST MOVE loaded: %d rows, %d columns", result.height, result.width
    )
    return result


# ===================================================================
# Loader: DREAMT (PhysioNet)
# ===================================================================


def load_dreamt(
    data_dir: str = "data",
) -> pl.DataFrame | None:
    """Load DREAMT dataset from PhysioNet (requires accepted terms).

    Expected columns: ``user_id, night, timestamp, accel_x/y/z, heart_rate, eda,
    skin_temperature, bvp, sleep_stage``.
    """
    dest = _physionet_download("dreamt/2.2.0", data_dir, "dreamt")
    if dest is None:
        return None

    csv_files = _find_csv_files(dest)
    tsv_files = _find_tsv_files(dest)
    all_files = csv_files + tsv_files
    if not all_files:
        logger.warning("No CSV/TSV files found in DREAMT download")
        return None

    frames: list[pl.DataFrame] = []
    for f in all_files:
        try:
            sep = "," if f.suffix == ".csv" else "\t"
            df = pl.read_csv(f, separator=sep, try_parse_dates=True)
            if df.width > 1:
                frames.append(df)
        except Exception:
            logger.debug("Skipping unreadable file %s", f)

    if not frames:
        return None
    result = pl.concat(frames, how="diagonal")
    for ts_col in ("date", "time", "timestamp", "datetime"):
        if ts_col in result.columns:
            result = result.rename({ts_col: "timestamp"})
            break
    if "timestamp" in result.columns and result["timestamp"].dtype != pl.Datetime:
        result = result.with_columns(
            result["timestamp"].cast(pl.Datetime, strict=False)
        )
    logger.info("DREAMT loaded: %d rows, %d columns", result.height, result.width)
    return result


# ===================================================================
# Loader: WESAD (Kaggle)
# ===================================================================


def load_wesad(
    data_dir: str = "data",
) -> pl.DataFrame | None:
    """Load WESAD wearable stress-affect detection dataset from Kaggle.

    Expected columns: ``user_id, timestamp, chest_ecg, chest_eda, chest_emg,
    chest_resp, chest_temp, wrist_accel_x/y/z, wrist_eda, wrist_temp, wrist_bvp,
    label``.
    """
    dest = _kaggle_download(
        "orvile/wesad-wearable-stress-affect-detection-dataset", data_dir, "wesad"
    )
    if dest is None:
        return None

    # WESAD data is stored in pickle files: WESAD/S2/S2.pkl, WESAD/S5/S5.pkl, etc.
    pkl_files = sorted(dest.rglob("*.pkl"))
    if not pkl_files:
        logger.warning("No .pkl files found in WESAD download")
        return None

    import pickle as _pickle

    frames: list[pl.DataFrame] = []
    for f in pkl_files:
        try:
            with open(f, "rb") as fh:
                raw = _pickle.load(fh, encoding="latin1")
            subject = raw["subject"]
            labels = raw["label"].flatten()
            chest = raw["signal"]["chest"]

            n = len(labels)
            data: dict[str, Any] = {
                "user_id": subject,
                "timestamp": range(n),
                "label": labels,
            }
            # Map chest signal keys to output column names
            col_map = {
                "acc": "chest_acc",
                "ecg": "chest_ecg",
                "eda": "chest_eda",
                "emg": "chest_emg",
                "temp": "chest_temp",
                "resp": "chest_resp",
            }
            for key, arr in chest.items():
                k = key.lower()
                col = col_map.get(k)
                if col is None:
                    continue
                if arr.ndim == 2 and arr.shape[1] == 3:
                    data[f"{col}_x"] = arr[:, 0]
                    data[f"{col}_y"] = arr[:, 1]
                    data[f"{col}_z"] = arr[:, 2]
                else:
                    data[col] = arr[:n, 0] if arr.ndim == 2 else arr[:n]

            df = pl.DataFrame(data)
            frames.append(df)
        except Exception:
            logger.debug("Skipping unreadable pickle %s", f)

    if not frames:
        return None
    result = pl.concat(frames, how="diagonal")
    result = result.with_columns(result["user_id"].cast(pl.Utf8))
    logger.info("WESAD loaded: %d rows, %d columns", result.height, result.width)
    return result


# ===================================================================
# Loader: WISDM (Direct download)
# ===================================================================

_WISDM_URL = (
    "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
)


def load_wisdm(
    data_dir: str = "data",
) -> pl.DataFrame | None:
    """Load WISDM activity-recognition dataset (direct download).

    Expected columns: ``user_id, timestamp, activity, accelerometer_x/y/z``.
    """
    raw_dir = _ensure_dir(Path(data_dir) / "wisdm")
    raw_txt = raw_dir / "WISDM_ar_v1.1_raw.txt"
    marker = raw_dir / ".wisdm_done"

    if marker.exists() and raw_txt.exists():
        logger.info("WISDM already cached at %s", raw_txt)
    else:
        # Download tar.gz and extract
        tar_path = raw_dir / "WISDM_ar_latest.tar.gz"
        try:
            _download_file(_WISDM_URL, tar_path)
        except Exception:
            logger.exception("Failed to download WISDM dataset")
            return None

        # Extract the raw txt file
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("_raw.txt"):
                        tar.extract(member, path=raw_dir)
                        # Move to expected location
                        extracted = raw_dir / member.name
                        if extracted != raw_txt and not raw_txt.exists():
                            extracted.rename(raw_txt)
                        break
            tar_path.unlink()  # clean up
            marker.touch()
        except Exception:
            logger.exception("Failed to extract WISDM data")
            return None

    if not raw_txt.exists():
        return None

    # WISDM raw file is comma-separated, no header, trailing semicolons.
    # Each line looks like: "33,Jogging,49105962326000,-0.6946377,12.680544,0.50395286;"
    # Read as text, strip trailing semicolons, then parse as CSV.
    # Columns: user_id, activity, timestamp, accelerometer_x, accelerometer_y, accelerometer_z
    try:
        raw_text = raw_txt.read_text(encoding="utf-8")
        # Some lines contain two records concatenated with a semicolon.
        # Split on semicolons to get individual records, then filter empties.
        records = [r.strip() for r in raw_text.replace("\n", ";").split(";")]
        records = [r for r in records if r]
        cleaned = "\n".join(records)

        df = pl.read_csv(
            StringIO(cleaned),
            separator=",",
            has_header=False,
            new_columns=[
                "user_id",
                "activity",
                "timestamp",
                "accelerometer_x",
                "accelerometer_y",
                "accelerometer_z",
            ],
            truncate_ragged_lines=True,
            try_parse_dates=False,
        )
        df = df.with_columns(
            pl.from_epoch(df["timestamp"].cast(pl.Int64), time_unit="ms")
        )
        df = df.with_columns(df["user_id"].cast(pl.Utf8))
        logger.info("WISDM loaded: %d rows, %d columns", df.height, df.width)
        return df
    except Exception:
        logger.exception("Failed to parse WISDM data file")
        return None


# ===================================================================
# Loader: MHEALTH (UCI ML Repo)
# ===================================================================


def load_mhealth(
    data_dir: str = "data",
) -> pl.DataFrame | None:
    """Load MHEALTH dataset from the UCI Machine Learning Repository.

    Expected columns: ``user_id, timestamp, chest_accel_x/y/z, chest_gyro_x/y/z,
    chest_mag_x/y/z, chest_ecg, wrist_accel_x/y/z, ankle_accel_x/y/z,
    activity``.
    """
    cache_dir = _ensure_dir(Path(data_dir) / "mhealth")
    marker = cache_dir / ".mhealth_done"
    if marker.exists():
        logger.info("MHEALTH already cached at %s", cache_dir)

    try:
        from ucimlrepo import fetch_ucirepo  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        logger.warning("ucimlrepo not installed; cannot load MHEALTH")
        return None

    try:
        repo = fetch_ucirepo(id=319)
    except Exception:
        logger.exception("Failed to fetch MHEALTH dataset from UCI")
        return None

    # ucimlrepo returns pandas DataFrames
    if repo.data is None or repo.data.features is None:
        logger.warning("MHEALTH: no features returned from UCI")
        return None

    pdf = repo.data.features
    if repo.data.targets is not None:
        pdf["activity"] = repo.data.targets.values.ravel()

    # Add user_id column derived from the original dataset
    # MHEALTH data is structured with 10 subjects
    # The raw CSV files each represent one subject
    # We'll try to infer user_id from an existing column or set a placeholder
    if "subject" in pdf.columns:
        pdf.rename(columns={"subject": "user_id"}, inplace=True)

    df = pl.from_pandas(pdf)
    # Ensure standard columns
    if "user_id" not in df.columns:
        df = df.with_columns(pl.lit(0).alias("user_id"))
    for ts_col in ("date", "time", "timestamp", "datetime"):
        if ts_col in df.columns:
            df = df.rename({ts_col: "timestamp"})
            break

    marker.touch()
    logger.info("MHEALTH loaded: %d rows, %d columns", df.height, df.width)
    return df


# ===================================================================
# Loader: ExtraSensory (Direct download)
# ===================================================================

_EXTRASENSORY_URLS = [
    "http://extrasensory.ucsd.edu/extra_sensory_data/",
    "https://extrasensory.ucsd.edu/extra_sensory_data/",
]


def load_extrasensory(
    data_dir: str = "data",
) -> pl.DataFrame | None:
    """Load ExtraSensory dataset from UCSD (direct download).

    Expected columns: ``user_id, timestamp, activity, accelerometer_x/y/z,
    gyroscope_x/y/z, compass_x/y/z, audio_level, location_lat/lon, mood,
    social_context, phone_usage``.

    .. note::
       The ExtraSensory website was intermittently unavailable as of mid-2025.
       This loader will return ``None`` if the server is unreachable.
    """
    dest_dir = _ensure_dir(Path(data_dir) / "extrasensory")
    marker = dest_dir / ".extrasensory_done"
    if marker.exists():
        logger.info("ExtraSensory already cached at %s", dest_dir)
    else:
        # Try known download URLs
        downloaded = False
        for base_url in _EXTRASENSORY_URLS:
            for fname in ("extra_sensory_data.zip", "extra_sensory_data.csv"):
                url = urljoin(base_url, fname)
                try:
                    resp = requests.head(url, timeout=10, allow_redirects=True)
                    if resp.status_code == 200:
                        dest = dest_dir / fname
                        _download_file(url, dest)
                        downloaded = True
                        break
                except requests.RequestException:
                    continue
            if downloaded:
                break
        if not downloaded:
            logger.warning(
                "ExtraSensory dataset unavailable (server may be down). Skipping."
            )
            return None

        # Extract zip if needed
        zip_path = dest_dir / "extra_sensory_data.zip"
        if zip_path.exists():
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(dest_dir)
                zip_path.unlink()
            except Exception:
                logger.exception("Failed to extract ExtraSensory zip")
                return None

        marker.touch()

    csv_files = _find_csv_files(dest_dir)
    tsv_files = _find_tsv_files(dest_dir)
    all_files = csv_files + tsv_files
    if not all_files:
        logger.warning("No data files found in ExtraSensory download")
        return None

    frames: list[pl.DataFrame] = []
    for f in all_files:
        try:
            sep = "," if f.suffix == ".csv" else "\t"
            df = pl.read_csv(f, separator=sep, try_parse_dates=True)
            if df.width > 1:
                frames.append(df)
        except Exception:
            logger.debug("Skipping unreadable file %s", f)

    if not frames:
        return None
    result = pl.concat(frames, how="diagonal")
    for ts_col in ("date", "time", "timestamp", "datetime"):
        if ts_col in result.columns:
            result = result.rename({ts_col: "timestamp"})
            break
    if "timestamp" in result.columns and result["timestamp"].dtype != pl.Datetime:
        result = result.with_columns(
            result["timestamp"].cast(pl.Datetime, strict=False)
        )

    # Ensure user_id is string
    if "user_id" in result.columns:
        result = result.with_columns(result["user_id"].cast(pl.Utf8))

    logger.info("ExtraSensory loaded: %d rows, %d columns", result.height, result.width)
    return result


# ===================================================================
# Loader: Fitbit Tracker (Kaggle)
# ===================================================================


def load_fitbit_tracker(
    data_dir: str = "data",
) -> pl.DataFrame | None:
    """Load Fitbit Fitness Tracker dataset from Kaggle.

    Expected columns: ``user_id, date, step_count, heart_rate, sleep_minutes,
    sleep_efficiency, calories, distance, active_minutes, sedentary_minutes``.
    """
    dest = _kaggle_download(
        "haseeb85/fitbit-fitness-tracker-data", data_dir, "fitbit_tracker"
    )
    if dest is None:
        return None

    csv_files = _find_csv_files(dest)
    if not csv_files:
        logger.warning("No CSV files found in Fitbit download")
        return None

    # Only load dailyActivity_merged.csv — it has the richest daily-level data.
    daily = [f for f in csv_files if f.name == "dailyActivity_merged.csv"]
    if not daily:
        logger.warning("dailyActivity_merged.csv not found in Fitbit download")
        return None

    try:
        df = pl.read_csv(
            daily[0],
            try_parse_dates=True,
            truncate_ragged_lines=True,
            infer_schema_length=10000,
        )
    except Exception:
        logger.exception("Failed to parse dailyActivity_merged.csv")
        return None

    # Rename Id -> user_id and standardise columns
    if "Id" in df.columns:
        df = df.rename({"Id": "user_id"})
    for ts_col in ("ActivityDate", "date", "timestamp", "time", "datetime"):
        if ts_col in df.columns:
            df = df.rename({ts_col: "timestamp"})
            break
    if "timestamp" in df.columns and df["timestamp"].dtype != pl.Datetime:
        df = df.with_columns(df["timestamp"].cast(pl.Datetime, strict=False))
    if "user_id" in df.columns:
        df = df.with_columns(df["user_id"].cast(pl.Utf8))

    logger.info(
        "Fitbit Tracker loaded: %d rows, %d columns", df.height, df.width
    )
    return df


# ===================================================================
# Loader: 4TU Step Goals (Direct download)
# ===================================================================

_4TU_FILE_URL = (
    "https://data.4tu.nl/file/6f8e6750-7494-4226-b6f9-299a9edbb077/"
    "2cc76f48-4eaa-4b06-877a-e9c885b6d1e9"
)


def load_4tu_step_goals(
    data_dir: str = "data",
) -> pl.DataFrame | None:
    """Load the 4TU RL-based step goals dataset (direct download).

    Expected columns: ``user_id, date, step_count, assigned_goal, goal_met,
    intervention_type, day_of_week``.
    """
    dest_dir = _ensure_dir(Path(data_dir) / "4tu_step_goals")
    marker = dest_dir / ".4tu_done"
    main_csv = dest_dir / "cleaned_database_data.csv"

    if marker.exists() and main_csv.exists():
        logger.info("4TU Step Goals already cached at %s", main_csv)
    else:
        zip_path = dest_dir / "4tu_data.zip"
        try:
            _download_file(_4TU_FILE_URL, zip_path)
        except Exception:
            logger.exception("Failed to download 4TU Step Goals dataset")
            return None

        # Extract the relevant CSV files
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    if member.endswith(".csv"):
                        zf.extract(member, dest_dir)
            zip_path.unlink()

            # Move the CSV up one level for easier access
            for csv_file in dest_dir.rglob("*.csv"):
                if csv_file != main_csv:
                    try:
                        csv_file.rename(main_csv)
                    except FileExistsError:
                        pass
            marker.touch()
        except Exception:
            logger.exception("Failed to extract 4TU data")
            return None

    if not main_csv.exists():
        # Maybe CSV is in a subdirectory
        csv_files = _find_csv_files(dest_dir)
        if csv_files:
            main_csv = csv_files[0]
        else:
            return None

    try:
        df = pl.read_csv(main_csv, try_parse_dates=True)
    except Exception:
        logger.exception("Failed to parse 4TU CSV")
        return None

    for ts_col in ("date", "timestamp", "time", "datetime"):
        if ts_col in df.columns:
            df = df.rename({ts_col: "timestamp"})
            break
    if "timestamp" in df.columns and df["timestamp"].dtype != pl.Datetime:
        df = df.with_columns(df["timestamp"].cast(pl.Datetime, strict=False))
    if "user_id" in df.columns:
        df = df.with_columns(df["user_id"].cast(pl.Utf8))

    logger.info("4TU Step Goals loaded: %d rows, %d columns", df.height, df.width)
    return df


# ===================================================================
# Loader: Synthetic
# ===================================================================


def load_synthetic(
    data_dir: str = "data",
    n_users: int = 100,
    n_timesteps: int = 365,
) -> pl.DataFrame | None:
    """Generate and return synthetic accelerometer data.

    Uses the existing :class:`rl_health_interventions.data.synthetic.SyntheticDataGenerator`
    and converts the result to a polars DataFrame with columns
    ``user_id, timestep, steps``.

    Parameters
    ----------
    data_dir:
        Ignored for synthetic data (used only for consistency).
    n_users:
        Number of synthetic users.
    n_timesteps:
        Number of timesteps per user.
    """
    from rl_health_interventions.data.synthetic import SyntheticDataGenerator

    generator = SyntheticDataGenerator(seed=42)
    ds = generator.generate(n_users=n_users, n_timesteps=n_timesteps)

    # Convert the Dataset (numpy-based) to a tidy polars DataFrame
    data: dict[str, Any] = {
        "user_id": np.repeat(ds.user_ids, n_timesteps),
        "timestep": np.tile(np.arange(n_timesteps), n_users),
    }
    for feat_name, feat_arr in ds.features.items():
        data[feat_name] = feat_arr.flatten()

    df = pl.DataFrame(data)
    df = df.with_columns(df["user_id"].cast(pl.Utf8))

    logger.info("Synthetic data generated: %d rows, %d columns", df.height, df.width)
    return df


# ===================================================================
# Registry and bulk loader
# ===================================================================


def get_all_loaders() -> dict[str, Callable[..., pl.DataFrame | None]]:
    """Return a name-to-function mapping of all available dataset loaders."""
    return {
        "pmdata": load_pmdata,
        "harth": load_harth,
        "bidsleep": load_bidsleep,
        "scientisst_move": load_scientisst_move,
        "dreamt": load_dreamt,
        "wesad": load_wesad,
        "wisdm": load_wisdm,
        "mhealth": load_mhealth,
        "extrasensory": load_extrasensory,
        "fitbit_tracker": load_fitbit_tracker,
        "4tu_step_goals": load_4tu_step_goals,
        "synthetic": load_synthetic,
    }


def load_all(
    data_dir: str = "data",
) -> dict[str, pl.DataFrame | None]:
    """Call every registered loader and return name → DataFrame (or ``None``).

    Results are returned in a dict keyed by dataset name.  Any loader that fails
    (due to missing credentials, network errors, etc.) logs a warning and yields
    ``None`` for that key.
    """
    results: dict[str, pl.DataFrame | None] = {}
    loaders = get_all_loaders()
    for name, loader_fn in loaders.items():
        logger.info("=" * 60)
        logger.info("Loading dataset: %s", name)
        try:
            results[name] = loader_fn(data_dir=data_dir)
        except Exception:
            logger.exception("Unhandled error loading %s", name)
            results[name] = None

        status = "OK" if results[name] is not None else "FAILED/SKIPPED"
        logger.info("Dataset %s: %s", name, status)

    n_ok = sum(1 for v in results.values() if v is not None)
    n_total = len(results)
    logger.info("load_all finished: %d/%d datasets loaded successfully", n_ok, n_total)
    return results

"""
CHB-MIT EEG Dataset Download and Parsing Module

This module handles:
- Downloading EEG files from the CHB-MIT Scalp EEG Database
- Parsing seizure annotation summary files
- Managing dataset organization

Dataset: CHB-MIT Scalp EEG Database
Source: https://physionet.org/content/chbmit/1.0.0/
"""

import os
import re
import urllib.request
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# CHB-MIT PhysioNet base URL
PHYSIONET_BASE_URL = "https://physionet.org/files/chbmit/1.0.0"


@dataclass
class SeizureInfo:
    """Information about a single seizure event."""

    file_name: str
    start_time: int  # Seconds from file start
    end_time: int  # Seconds from file start

    @property
    def duration(self) -> int:
        """Duration of seizure in seconds."""
        return self.end_time - self.start_time


@dataclass
class PatientInfo:
    """Information about a patient's recording session."""

    patient_id: str
    files: List[str]
    seizures: List[SeizureInfo]
    sampling_rate: int = 256
    n_channels: int = 22


def download_file(url: str, destination: str, verbose: bool = True) -> bool:
    """
    Download a file from URL to destination.

    Args:
        url: Source URL
        destination: Local file path
        verbose: Print progress messages

    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(destination):
            if verbose:
                print(f"  File already exists: {os.path.basename(destination)}")
            return True

        os.makedirs(os.path.dirname(destination), exist_ok=True)

        if verbose:
            print(f"  Downloading: {os.path.basename(destination)}...")

        urllib.request.urlretrieve(url, destination)

        if verbose:
            print(f"  ✓ Downloaded: {os.path.basename(destination)}")
        return True

    except Exception as e:
        print(f"  ✗ Failed to download {url}: {e}")
        return False


def download_chb_mit_sample(
    patient_ids: List[str] = None,
    data_dir: str = None,
    max_files_per_patient: int = 5,
    verbose: bool = True,
) -> Dict[str, PatientInfo]:
    """
    Download a sample of the CHB-MIT EEG dataset.

    Args:
        patient_ids: List of patient IDs to download (e.g., ["chb01", "chb02"])
                    If None, downloads chb01, chb02, chb03
        data_dir: Directory to save data. If None, uses default from config.
        max_files_per_patient: Maximum number of EDF files per patient
        verbose: Print progress messages

    Returns:
        Dictionary mapping patient_id to PatientInfo
    """
    if patient_ids is None:
        patient_ids = ["chb01", "chb02", "chb03"]

    if data_dir is None:
        from config import DATA_DIR

        data_dir = DATA_DIR

    if verbose:
        print("=" * 60)
        print("CHB-MIT EEG Dataset Download")
        print("=" * 60)

    patients_info = {}

    for patient_id in patient_ids:
        if verbose:
            print(f"\nProcessing patient: {patient_id}")
            print("-" * 40)

        patient_dir = os.path.join(data_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        # Download summary file first
        summary_url = f"{PHYSIONET_BASE_URL}/{patient_id}/{patient_id}-summary.txt"
        summary_path = os.path.join(patient_dir, f"{patient_id}-summary.txt")

        if not download_file(summary_url, summary_path, verbose):
            continue

        # Parse summary to get file list and seizure info
        seizures, file_list = parse_seizure_summary(summary_path)

        # Download EDF files (limited to max_files_per_patient)
        downloaded_files = []
        seizures_downloaded = []

        # Prioritize files with seizures
        files_with_seizures = set(s.file_name for s in seizures)
        priority_files = [f for f in file_list if f in files_with_seizures]
        other_files = [f for f in file_list if f not in files_with_seizures]

        files_to_download = (priority_files + other_files)[:max_files_per_patient]

        for file_name in files_to_download:
            file_url = f"{PHYSIONET_BASE_URL}/{patient_id}/{file_name}"
            file_path = os.path.join(patient_dir, file_name)

            if download_file(file_url, file_path, verbose):
                downloaded_files.append(file_name)
                # Track seizures in downloaded files
                for s in seizures:
                    if s.file_name == file_name:
                        seizures_downloaded.append(s)

        patients_info[patient_id] = PatientInfo(
            patient_id=patient_id,
            files=downloaded_files,
            seizures=seizures_downloaded,
        )

        if verbose:
            print(f"  Downloaded {len(downloaded_files)} files")
            print(f"  Seizures in downloaded files: {len(seizures_downloaded)}")

    if verbose:
        print("\n" + "=" * 60)
        print("Download Complete!")
        total_files = sum(len(p.files) for p in patients_info.values())
        total_seizures = sum(len(p.seizures) for p in patients_info.values())
        print(f"Total files: {total_files}")
        print(f"Total seizures: {total_seizures}")
        print("=" * 60)

    return patients_info


def parse_seizure_summary(summary_path: str) -> Tuple[List[SeizureInfo], List[str]]:
    """
    Parse CHB-MIT summary file to extract seizure information.

    Args:
        summary_path: Path to the summary.txt file

    Returns:
        Tuple of (list of SeizureInfo, list of file names)
    """
    seizures = []
    files = []

    with open(summary_path, "r") as f:
        content = f.read()

    # Split by file entries
    file_blocks = re.split(r"File Name:", content)[1:]

    for block in file_blocks:
        lines = block.strip().split("\n")

        # Get file name
        file_name = lines[0].strip()
        files.append(file_name)

        # Check for seizures
        if "Number of Seizures in File: 0" in block:
            continue

        # Extract seizure times
        start_times = re.findall(r"Seizure.*Start Time:\s*(\d+)\s*seconds", block)
        end_times = re.findall(r"Seizure.*End Time:\s*(\d+)\s*seconds", block)

        for start, end in zip(start_times, end_times):
            seizures.append(
                SeizureInfo(
                    file_name=file_name, start_time=int(start), end_time=int(end)
                )
            )

    return seizures, files


def get_patient_summary(patients_info: Dict[str, PatientInfo]) -> str:
    """
    Generate a summary string for downloaded patient data.

    Args:
        patients_info: Dictionary from download_chb_mit_sample

    Returns:
        Formatted summary string
    """
    lines = ["CHB-MIT Dataset Summary", "=" * 40]

    for patient_id, info in patients_info.items():
        lines.append(f"\n{patient_id}:")
        lines.append(f"  Files: {len(info.files)}")
        lines.append(f"  Seizures: {len(info.seizures)}")

        for s in info.seizures:
            lines.append(
                f"    - {s.file_name}: {s.start_time}s - {s.end_time}s ({s.duration}s)"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    # Test download with a small sample
    patients = download_chb_mit_sample(
        patient_ids=["chb01"], max_files_per_patient=3, verbose=True
    )
    print("\n" + get_patient_summary(patients))

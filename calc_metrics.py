"""
Script to calculate WER (Word Error Rate) and CER (Character Error Rate)
given paths to ground truth and predicted transcriptions.

Usage:
    python3 calc_metrics.py --ground_truth_dir PATH_TO_GT --predicted_dir PATH_TO_PRED

    where:
    - PATH_TO_GT: directory containing ground truth transcription files (*.txt)
    - PATH_TO_PRED: directory containing predicted transcription files (*.txt)

Example:
    python3 calc_metrics.py \
        --ground_truth_dir data/ground_truth \
        --predicted_dir data/predictions
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

from src.metrics.utils import calc_cer, calc_wer

logger = logging.getLogger(__name__)


def load_transcriptions(directory: Path) -> Dict[str, str]:
    """
    Load transcription files from a directory.
    
    Args:
        directory (Path): Directory containing .txt files with transcriptions.
        
    Returns:
        dict: Dictionary mapping utterance IDs to transcription text.
    """
    transcriptions = {}
    txt_files = list(Path(directory).glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {directory}")
    
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            transcriptions[txt_file.stem] = f.read().strip()
    
    return transcriptions


def extract_transcription_text(content: str) -> str:
    """
    Extract transcription text from a file that may contain multiple lines.
    
    If the file contains "Predicted: ..." or "Target: ...", extract just the text.
    Otherwise, return the entire content.
    
    Args:
        content (str): File content
        
    Returns:
        str: Extracted transcription text
    """
    lines = content.split("\n")
    
    # Try to find "Predicted:" or "Target:" lines
    for line in lines:
        if line.startswith("Predicted:"):
            return line.replace("Predicted:", "").strip()
        elif line.startswith("Target:"):
            return line.replace("Target:", "").strip()
    
    # If no marker found, return first non-empty line
    for line in lines:
        if line.strip():
            return line.strip()
    
    return content.strip()


def calculate_metrics(
    ground_truth_dir: Path,
    predicted_dir: Path,
) -> Tuple[float, float, Dict]:
    """
    Calculate WER and CER metrics.
    
    Args:
        ground_truth_dir (Path): Directory with ground truth transcriptions.
        predicted_dir (Path): Directory with predicted transcriptions.
        
    Returns:
        tuple: (wer, cer, detailed_results)
            - wer (float): Word Error Rate (%)
            - cer (float): Character Error Rate (%)
            - detailed_results (dict): Detailed results per utterance
    """
    gt_transcriptions = load_transcriptions(ground_truth_dir)
    pred_transcriptions = load_transcriptions(predicted_dir)
    
    # Find common utterance IDs
    common_ids = set(gt_transcriptions.keys()) & set(pred_transcriptions.keys())
    
    if not common_ids:
        raise ValueError(
            f"No matching utterance IDs found between "
            f"{ground_truth_dir} and {predicted_dir}"
        )
    
    wers = []
    cers = []
    detailed_results = {}
    
    missing_in_pred = set(gt_transcriptions.keys()) - set(pred_transcriptions.keys())
    missing_in_gt = set(pred_transcriptions.keys()) - set(gt_transcriptions.keys())
    
    if missing_in_pred:
        logger.warning(
            f"Missing in predictions: {missing_in_pred} "
            f"({len(missing_in_pred)} utterances)"
        )
    
    if missing_in_gt:
        logger.warning(
            f"Extra in predictions: {missing_in_gt} "
            f"({len(missing_in_gt)} utterances)"
        )
    
    for utt_id in sorted(common_ids):
        gt_text = extract_transcription_text(gt_transcriptions[utt_id])
        pred_text = extract_transcription_text(pred_transcriptions[utt_id])
        
        wer = calc_wer(gt_text, pred_text)
        cer = calc_cer(gt_text, pred_text)
        
        wers.append(wer)
        cers.append(cer)
        
        detailed_results[utt_id] = {
            "wer": wer,
            "cer": cer,
            "ground_truth": gt_text,
            "predicted": pred_text,
        }
    
    avg_wer = sum(wers) / len(wers) * 100.0
    avg_cer = sum(cers) / len(cers) * 100.0
    
    return avg_wer, avg_cer, detailed_results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate WER and CER metrics from transcription directories."
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        required=True,
        help="Path to directory with ground truth transcriptions (.txt files)",
    )
    parser.add_argument(
        "--predicted_dir",
        type=str,
        required=True,
        help="Path to directory with predicted transcriptions (.txt files)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed results per utterance",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.INFO,
    )
    
    gt_dir = Path(args.ground_truth_dir)
    pred_dir = Path(args.predicted_dir)
    
    logger.info(f"Ground truth directory: {gt_dir}")
    logger.info(f"Predicted directory: {pred_dir}")
    
    try:
        avg_wer, avg_cer, detailed_results = calculate_metrics(gt_dir, pred_dir)
        
        print("\n" + "=" * 60)
        print("METRICS SUMMARY")
        print("=" * 60)
        print(f"Average WER (Word Error Rate): {avg_wer:.2f}%")
        print(f"Average CER (Character Error Rate): {avg_cer:.2f}%")
        print(f"Total utterances processed: {len(detailed_results)}")
        print("=" * 60 + "\n")
        
        if args.verbose:
            print("DETAILED RESULTS PER UTTERANCE:")
            print("-" * 60)
            for utt_id, results in sorted(detailed_results.items()):
                print(f"\nUtterance ID: {utt_id}")
                print(f"  WER: {results['wer']:.2f}%")
                print(f"  CER: {results['cer']:.2f}%")
                print(f"  Ground Truth: {results['ground_truth'][:80]}")
                print(f"  Predicted:    {results['predicted'][:80]}")
            print("-" * 60 + "\n")
    
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    batch = {}

    audios = [item["audio"] for item in dataset_items]
    audio_lens = [audio.shape[-1] for audio in audios]
    max_audio_len = max(audio_lens)
    batch["audio"] = torch.stack([
        torch.nn.functional.pad(audio, (0, max_audio_len - audio.shape[-1]))
        for audio in audios
    ])
    batch["audio_length"] = torch.tensor(audio_lens)
    batch["audio_path"] = [item["audio_path"] for item in dataset_items]

    specs = [item["spectrogram"] for item in dataset_items]
    spec_lens = [spec.shape[-1] for spec in specs]
    max_spec_len = max(spec_lens)
    batch["spectrogram"] = torch.stack([
        torch.nn.functional.pad(spec, (0, max_spec_len - spec.shape[-1]))
        for spec in specs
    ])
    batch["spectrogram_length"] = torch.tensor(spec_lens)

    text_encoded = [item["text_encoded"].squeeze(0) for item in dataset_items]
    text_lens = [t.shape[-1] for t in text_encoded]
    max_text_len = max(text_lens)
    batch["text_encoded"] = torch.stack([
        torch.nn.functional.pad(t, (0, max_text_len - t.shape[-1]), value=0)
        for t in text_encoded
    ])
    batch["text_encoded_length"] = torch.tensor(text_lens)

    batch["text"] = [item["text"] for item in dataset_items]

    batch["audio_orig"] = [item["audio_orig"] for item in dataset_items]
    batch["spectrogram_orig"] = [item["spectrogram_orig"] for item in dataset_items]

    return batch

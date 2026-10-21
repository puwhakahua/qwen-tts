import argparse
import os.path

import torch
import soundfile as sf
import traceback
from pathlib import Path
from qwen_tts import Qwen3TTSModel


def clone(ref_audio: str, ref_text: str, input_file: str, output_dir: str, language: str = "English",
          model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device: str = "cuda:0",
          save_text: bool = False):
    """
    Clones the voice specified by the reference audio and text and applies it to the lines of text
    in the specified input file. The generate audio is stored in the specified output dir.

    :param ref_audio: the WAV file with the 30s of reference audio to use
    :type ref_audio: str
    :param ref_text: the transcript of the reference audio
    :type ref_text: str
    :param input_file: path to the text file with the lines of text to generate audio from
    :type input_file: str
    :param output_dir: the directory to store the generated output in
    :type output_dir: str
    :param language: the language to use
    :type language: str
    :param model: the model to use
    :type model: str
    :param device: the device to use, eg cuda:0
    :type device: str
    :param save_text: whether to save the text alongside the generated audio file
    :type save_text: bool
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Loading lines of text from: %s" % input_file)
    with open(input_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    total = len(texts)
    print("%d lines loaded." % total)

    # load model 1.7B Base
    print("Loading model: %s" % model)
    model = Qwen3TTSModel.from_pretrained(
        model,
        device_map=device,
        dtype=torch.bfloat16,
    )
    print("Model loaded.")

    # generate
    for i, text in enumerate(texts, 1):
        out_file = out_path / f"{i:05d}.wav"
        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                language=language,
            )

            sf.write(str(out_file), wavs[0], sr)
            if save_text:
                out_text = os.path.splitext(str(out_file))[0] + ".txt"
                with open(out_text, "w") as fp:
                    fp.write(text)
                    fp.write("\n")

            if i % 100 == 0:
                print("%d/%d..." % (i+1, total))

        except Exception:
            print("Failed to generate audio for: %s" % text)
            traceback.print_exc()
            continue

    print("Finished!")


def main():
    parser = argparse.ArgumentParser(description="qwen-tts - voice cloning", prog="qwen_tts_clone", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ref_audio', help='Path to the 30s of reference audio to use', required=True, default=None)
    parser.add_argument('--ref_text', help='The transcript of the reference audio', required=True, default=None)
    parser.add_argument('--input_file', help='The text file with lines of text to generate audio for.', required=True, default=None)
    parser.add_argument('--output_dir', help='The directory to store the generated audio in.', required=True, default=None)
    parser.add_argument('--language', help='The language of the audio.', required=False, default="English")
    parser.add_argument('--model', help='The Qwen model to use.', required=False, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument('--device', help='The device to run the model on.', required=False, default="cuda:0")
    parser.add_argument('--save_text', action='store_true', help='Whether to save the text alongside the generated audio as well.', required=False)
    parsed = parser.parse_args()

    clone(parsed.ref_audio, parsed.ref_text, parsed.input_file, parsed.output_dir, language=parsed.language,
          model=parsed.model, device=parsed.device, save_text=parsed.save_text)


if __name__ == "__main__":
    main()

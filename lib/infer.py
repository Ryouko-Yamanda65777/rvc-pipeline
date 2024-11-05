
import os
import gc
from .config import Configs
from .utils import get_model
from lib.modules import VC  # Ensure this is correctly imported from your dependencies
from lib.split_audio import split_silence_nonsilent, adjust_audio_lengths, combine_silence_nonsilent

def infer_audio(model_name, audio_path, **kwargs):
    # Example of parameter unpacking for readability
    configs = Configs(kwargs.get('device', 'cuda:0'), kwargs.get('is_half', True))
    vc = VC(configs)
    pth_path, index_path = get_model(model_name)
    vc_data = vc.get_vc(pth_path, kwargs.get("protect", 0.33), 0.5)

    if kwargs.get("split_infer", False):
        temp_dir = os.path.join(os.getcwd(), "separate", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        silence_files, nonsilent_files = split_silence_nonsilent(
            audio_path, kwargs.get("min_silence", 500), kwargs.get("silence_threshold", -50),
            kwargs.get("seek_step", 1), kwargs.get("keep_silence", 100)
        )
        inferred_files = []
        for nonsilent_file in nonsilent_files:
            inference_info, audio_data, output_path = vc.vc_single(
                0, nonsilent_file, kwargs.get("f0_change", 0), kwargs.get("f0_method", "rmvpe+"),
                index_path, index_path, kwargs.get("index_rate", 0.75), kwargs.get("filter_radius", 3),
                kwargs.get("resample_sr", 0), kwargs.get("rms_mix_rate", 0.25), kwargs.get("protect", 0.33),
                kwargs.get("audio_format", "wav"), kwargs.get("crepe_hop_length", 128), kwargs.get("do_formant", False),
                kwargs.get("quefrency", 0), kwargs.get("timbre", 1), kwargs.get("min_pitch", "50"),
                kwargs.get("max_pitch", "1100"), kwargs.get("f0_autotune", False), kwargs.get("hubert_model_path", "assets/hubert/hubert_base.pt")
            )
            if inference_info[0] == "Success.":
                inferred_files.append(output_path)

        adjusted_inferred_files = adjust_audio_lengths(nonsilent_files, inferred_files)
        output_path = combine_silence_nonsilent(silence_files, adjusted_inferred_files, kwargs.get("keep_silence", 100))
    else:
        inference_info, audio_data, output_path = vc.vc_single(
            0, audio_path, kwargs.get("f0_change", 0), kwargs.get("f0_method", "rmvpe+"),
            index_path, index_path, kwargs.get("index_rate", 0.75), kwargs.get("filter_radius", 3),
            kwargs.get("resample_sr", 0), kwargs.get("rms_mix_rate", 0.25), kwargs.get("protect", 0.33),
            kwargs.get("audio_format", "wav"), kwargs.get("crepe_hop_length", 128), kwargs.get("do_formant", False),
            kwargs.get("quefrency", 0), kwargs.get("timbre", 1), kwargs.get("min_pitch", "50"),
            kwargs.get("max_pitch", "1100"), kwargs.get("f0_autotune", False), kwargs.get("hubert_model_path", "assets/hubert/hubert_base.pt")
        )

    del configs, vc
    gc.collect()
    return output_path

from .algorithms import gram_fusion_cpu, gram_fusion_gpu, high_pass_fusion_cpu, high_pass_fusion_gpu, hpf_fusion_cpu,\
    hpf_fusion_gpu, mean_value_fusion_cpu, mean_value_fusion_gpu
from .utils import utils


METHODS_FUSION = {
                    'hpf': {'cpu': hpf_fusion_cpu.fusion_hpf_cpu, 'gpu': hpf_fusion_gpu.fusion_hpf_gpu},
                    'mean_value': {'cpu': mean_value_fusion_cpu.fusion_valor_medio_cpu, 'gpu': mean_value_fusion_gpu.fusion_valor_medio_gpu},
                    'high_pass': {'cpu': high_pass_fusion_cpu.fusion_paso_alto_cpu, 'gpu':high_pass_fusion_gpu.fusion_paso_alto_gpu},
                    'gram': {'cpu': gram_fusion_cpu.fusion_gram_cpu, 'gpu': gram_fusion_gpu.fusion_gram_gpu}
                  }


def generate_fusion_images(multispectral_path, pancromatic_path, method_fusion, fusioned_image_path, device_fusion="cpu", geographical_info=True):
    fusion_algorithm = METHODS_FUSION[method_fusion][device_fusion]
    multi_image, multi_info = utils.read_image(multispectral_path)
    pan_image, pan_info = utils.read_image(pancromatic_path)
    image_fusioned = fusion_algorithm(multi_image, pan_image)
    if geographical_info:
        utils.save_image_with_info(fusioned_image_path, image_fusioned, multi_info)
    else:
        utils.save_image_without_info(fusioned_image_path, image_fusioned)

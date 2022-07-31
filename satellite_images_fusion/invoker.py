from satellite_images_fusion.algorithms import gram_fusion_cpu, gram_fusion_gpu, high_frecuency_modulation_cpu, high_frecuency_modulation_gpu, hpf_fusion_cpu,\
    hpf_fusion_gpu, mean_value_fusion_cpu, mean_value_fusion_gpu
from satellite_images_fusion.utils import utils
from satellite_images_fusion.metrics import metrics as mt


METHODS_FUSION = {
                    'hpf': {'cpu': hpf_fusion_cpu.fusion_hpf_cpu, 'gpu': hpf_fusion_gpu.fusion_hpf_gpu},
                    'mean_value': {'cpu': mean_value_fusion_cpu.fusion_mean_value_cpu, 'gpu': mean_value_fusion_gpu.fusion_mean_value_gpu},
                    'high_pass': {'cpu': high_frecuency_modulation_cpu.fusion_high_pass_cpu, 'gpu': high_frecuency_modulation_gpu.fusion_high_pass_gpu},
                    'gram': {'cpu': gram_fusion_cpu.fusion_gram_cpu, 'gpu': gram_fusion_gpu.fusion_gram_gpu}
                  }

METRICS_METHODS = {'mse': mt.mse,
                   'rmse': mt.rmse,
                   'bias': mt.bias,
                   'correlation': mt.correlation_coeff}


def generate_fusion_images(multispectral_path, pancromatic_path, method_fusion, fusioned_image_path, device_fusion="cpu", geographical_info=True):
    fusion_algorithm = METHODS_FUSION[method_fusion][device_fusion]
    multi_image, multi_info = utils.read_image(multispectral_path)
    pan_image, pan_info = utils.read_image(pancromatic_path)
    image_fusioned = fusion_algorithm(multi_image, pan_image)
    if geographical_info:
        utils.save_image_with_info(fusioned_image_path, image_fusioned, multi_info)
    else:
        utils.save_image_without_info(fusioned_image_path, image_fusioned)


def generate_quality_metrics(fusioned_image_path, original_image_path, metrics=['mse', 'rmse', 'bias', 'correlation']):
    results = {}
    fusioned_image, _ = utils.read_image(fusioned_image_path)
    original_image, _ = utils.read_image(original_image_path)
    for metric in metrics:
        if metric in METRICS_METHODS.keys():
            results[metric] = METRICS_METHODS[metric](fusioned_image, original_image)
        else:
            print(f"Metric {metric} is not defined")
    return results

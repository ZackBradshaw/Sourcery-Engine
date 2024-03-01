import gradio as gr
import sky

def deploy_vllm_on_sky(model_path, gpu_type, cpus, memory, cloud_provider, region, disk_size, disk_type):
    task = sky.Task(
        name="vllm_serving",
        setup="pip install vllm",
        run=f"vllm serve --model_name_or_path {model_path} --port 8080",
        envs={"MODEL_PATH": model_path},
        workdir=".",
        ports=8080
    )

    task.set_resources(
        sky.Resources(
            cloud=sky.Cloud(provider=cloud_provider, region=region),
            accelerators=f"{gpu_type}:1",
            cpus=cpus,
            memory=memory,
            disk=sky.Disk(size=disk_size, type=disk_type)
        )
    )

    cluster = sky.Cluster(
        name="vllm-cluster",
        cloud=sky.Cloud(provider=cloud_provider, region=region)
    )

    sky.launch(task, cluster=cluster)
    return f"VLLM model deployed on SkyPilot with cluster name: {cluster.name}"

sky_pilot_interface = gr.Interface(
    fn=deploy_vllm_on_sky,
    inputs=[
        gr.Textbox(label="Model Path", placeholder="EleutherAI/gpt-neo-2.7B"),
        gr.Dropdown(label="GPU Type", choices=["V100", "P100", "T4"], value="V100"),
        gr.Slider(label="CPUs", minimum=1, maximum=16, value=4),
        gr.Slider(label="Memory (GB)", minimum=4, maximum=64, value=16),
        gr.Dropdown(label="Cloud Provider", choices=["AWS", "GCP", "Azure"], value="AWS"),
        gr.Textbox(label="Region", placeholder="us-west-2"),
        gr.Slider(label="Disk Size (GB)", minimum=20, maximum=1000, value=100),
        gr.Dropdown(label="Disk Type", choices=["standard", "ssd"], value="ssd")
    ],
    outputs="text",
    title="Deploy VLLM on SkyPilot",
    description="Configure and deploy a VLLM model on a SkyPilot-managed cloud instance with full parameter customization."
)

if __name__ == "__main__":
    sky_pilot_interface.launch()
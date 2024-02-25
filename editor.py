import os
import warnings
import signal
import time
from pathlib import Path
import json
import yaml
from threading import Lock

import gradio as gr
from Fooocus.server import create_text_generation_interface
from text-generation-webui.webui import create_image_generation_interface

def main():
    logger.info("Starting Combined Interface")

    # Load custom settings
    settings_file = None
    if args_manager.args.settings is not None and Path(args_manager.args.settings).exists():
        settings_file = Path(args_manager.args.settings)
    elif Path('settings.yaml').exists():
        settings_file = Path('settings.yaml')
    elif Path('settings.json').exists():
        settings_file = Path('settings.json')

    if settings_file is not None:
        logger.info(f"Loading settings from \"{settings_file}\"")
        file_contents = open(settings_file, 'r', encoding='utf-8').read()
        new_settings = json.loads(file_contents) if settings_file.suffix == "json" else yaml.safe_load(file_contents)
        shared.settings.update(new_settings)

    # Fallback settings for models
    shared.model_config['.*'] = get_fallback_settings()
    shared.model_config.move_to_end('.*', last=False)  # Move to the beginning

    # Activate the extensions listed on settings.yaml
    extensions_module.available_extensions = utils.get_available_extensions()
    for extension in shared.settings['default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)

    available_models = utils.get_available_models()

    # Model selection logic

    # If any model has been selected, load it
    if shared.model_name != 'None':
        p = Path(shared.model_name)
        if p.exists():
            model_name = p.parts[-1]
            shared.model_name = model_name
        else:
            model_name = shared.model_name

        model_settings = get_model_metadata(model_name)
        update_model_parameters(model_settings, initial=True)

        # Load the model
        shared.model, shared.tokenizer = load_model(model_name)
        if shared.args.lora:
            add_lora_to_model(shared.args.lora)

    shared.generation_lock = Lock()

    if shared.args.nowebui:
        # Start the API in standalone mode
        shared.args.extensions = [x for x in shared.args.extensions if x != 'gallery']
        if shared.args.extensions is not None and len(shared.args.extensions) > 0:
            extensions_module.load_extensions()
    else:
        # Create Gradio interfaces using tabs
        with gr.Blocks(title="Combined Interface") as interface:
            with gr.Tab() as tab:
                with gr.TabItem(label="Text Generation"):
                    create_text_generation_interface()
                with gr.TabItem(label="Image Generation"):
                    create_image_generation_interface()

if __name__ == "__main__":
    main()
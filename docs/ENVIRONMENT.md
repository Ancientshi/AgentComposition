# Environment and paths

The recorded critic runtime used Python 3.11.15, PyTorch 2.7.1 with CUDA 12.6, Transformers 4.56.1, and an NVIDIA A800 80 GB. The generator adapter records PEFT 0.18.0. The requirement files list the project dependencies.

Set `BASE_MODEL`, `EASYREC_MODEL`, `CRITIC_URL`, `OPENAI_BASE_URL`, `BACKBONE_BASE_URL`, and `NATIVE_TOOL_URL` using `.env.example` as a reference. Export these values to the process environment. Keep credentials in your local gateway or environment.

Run commands from the repository checkout. Some optional baseline integrations and checkpoint metadata contain machine-specific paths; configure those paths for the local runtime. The command catalogue sets the import search path for its entries.

Training and search use CUDA. Containerized native tasks also use the relevant benchmark environments and Docker. Configure model and API aliases for the available services.

FROM nvcr.io/nvidia/tritonserver:25.03-py3

# Install Python dependencies for HuggingFace transformers
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install transformers==4.50.1
RUN pip install accelerate
RUN pip install sentencepiece
RUN pip install protobuf
RUN pip install librosa
RUN pip install flash-attn --no-build-isolation
RUN pip install resampy
# Set working directory
WORKDIR /opt/tritonserver

# Set cache directory for transformers
ENV TRANSFORMERS_CACHE=/opt/tritonserver/cache
RUN mkdir -p /opt/tritonserver/cache

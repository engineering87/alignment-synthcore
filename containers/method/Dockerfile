# Image for jailed method runs (harness/jail.py). The tag is a hash of this file:
# change a version here and every runner rebuilds, so paired runs share one image.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.5.4 scipy==1.18.1 scikit-learn==1.9.1 \
 && pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && useradd --uid 10001 --no-create-home method
# No harness, testbed or results code is copied in. The method sees only /sbx and itself.
USER method

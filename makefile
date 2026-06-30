
ORIGIN?=lutringer_10.mp3
DURATION?=18.0
TAG?=00

start1 :
# Basic conversion
	python prepare_audio.py \
               --input my_recording.mp3 \
               --output prepared_audio.wav

start :
# With silence trimming and normalization
	python prepare_audio.py \
              --input $(ORIGIN) \
              --output prepared_$(TAG)_$(ORIGIN) \
              --normalize \
              --duration $(DURATION) \
              --target-peak 0.08
	python training_script.py \
              --reference-audio prepared_$(TAG)_$(ORIGIN) \
              --model-path voxtral-tts-weights \
              --num-epochs 5000 \
              --learning-rate 0.1 \
              --checkpoint-dir cp_$(TAG)_$(ORIGIN) \
              --device cuda # mps  # or cuda/cpu
	python codes_to_embeddings.py \
              --codes cp_$(TAG)_$(ORIGIN)/final_codes.pt \
              --embedding-weight voxtral-tts-weights/consolidated.safetensors \
              --output $(TAG)_$(ORIGIN)_voice_embedding.pt \
              --add-end-token

tout :
	-ORIGIN=lutringer_10.mp3 DURATION=10.0 make start
	-ORIGIN=noiret.wav DURATION=45.0 make start
	-ORIGIN=louis.wav DURATION=20.0 make start
	-ORIGIN=louis.wav TAG=40 DURATION=20.0 make start


.PHONY: data baselines reproduce judge reproduce-live gate test

data:
	python -m src.data --download

baselines:
	python -m eval.baselines

reproduce:
	python -m eval.run_offline --judge gpt4_pair

reproduce-live:
	python -m eval.run_offline --judge cached-live

judge:
	python -m src.judge --pairs 300 --both-orders

gate:
	python -m eval.gate

test:
	pytest -q

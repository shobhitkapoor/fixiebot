train:
	python3 scripts/train.py

train_bert:
	python3 scripts/train_bert_fix.py

predict:
	python3 scripts/predict.py

run:
	python3 api/app.py

run_bert:
	python3 api/app_bert.py

feedback:
	python3 scripts/feedback_loop.py

ui:
	streamlit run streamlit_ui/app.py

cluster:
	python3 reports/cluster_visualization.py

jira:
	python3 integrations/jira_webhook.py

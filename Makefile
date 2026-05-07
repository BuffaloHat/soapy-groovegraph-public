dashboard: ## Start Docker, Ollama, and Streamlit dashboard (one command)
	@bash scripts/start_sgg.sh
up:        ## Start Postgres and Qdrant (Docker only)
	docker compose -f infra_sgg/docker-compose.yml up -d
down:      ## Stop all Docker containers (containers preserved, use 'docker compose down' to remove)
	docker compose -f infra_sgg/docker-compose.yml stop
ps:        ## Show container status
	docker compose -f infra_sgg/docker-compose.yml ps
help:      ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

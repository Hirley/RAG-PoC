Feature: Command-line entrypoint

  As an engineer evaluating the pipeline
  I want to drive ingestion, retrieval and answering from the terminal
  So that I can exercise the RAG system end to end without writing code

  Scenario: Ingesting documents from a JSON file
    Given a JSON file holding 3 documents
    When I run the CLI with "ingest" against that file
    Then the exit code should be 0
    And the documents should have been sent to the index
    And the output should report 3 indexed documents

  Scenario: Inspecting retrieval without calling the LLM
    Given the search stage returns a document titled "Deployment schedule"
    When I run the CLI with "search When do deploys happen?"
    Then the exit code should be 0
    And the output should contain "Deployment schedule"
    And the LLM should not have been called

  Scenario: Asking a question end to end
    Given the API key is configured
    And the RAG pipeline answers "Deploys happen at 10 PM."
    When I run the CLI with "ask When do deploys happen?"
    Then the exit code should be 0
    And the pipeline should have received the question "When do deploys happen?"
    And the output should contain "Deploys happen at 10 PM."

  Scenario: Asking without an API key configured
    Given the API key is missing
    When I run the CLI with "ask When do deploys happen?"
    Then the exit code should be 1
    And the error output should mention "ANTHROPIC_API_KEY"
    And the pipeline should not have been called

  Scenario: Running against an unreachable ElasticSearch
    Given the API key is configured
    And ElasticSearch is unreachable
    When I run the CLI with "ask When do deploys happen?"
    Then the exit code should be 1
    And the error output should mention the ElasticSearch URL

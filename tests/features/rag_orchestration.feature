Feature: RAG orchestration

  Scenario: Answering a question end to end
    Given the search stage returns 2 relevant documents
    When I call the rag function with the question "What is RAG?"
    Then the search stage should be called with the original question
    And the prompt stage should receive the search results
    And the LLM stage should receive the built prompt
    And the returned answer should be the LLM response

  Scenario: No relevant documents found
    Given the search stage returns no documents
    When I call the rag function with the question "Who won the 1998 World Cup?"
    Then the prompt stage should still be called with an empty result list
    And the LLM stage should receive a prompt carrying the fallback instruction

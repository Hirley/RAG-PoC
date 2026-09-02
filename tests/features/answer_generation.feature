Feature: Answer generation (LLM)

  Scenario: Response based on the provided context
    Given the formatted prompt contains the information "The main server shuts down at 10 PM"
    When I send this prompt to the LLM via API
    Then the returned response should mention the 10 PM time

  Scenario: Question outside corporate scope
    Given the search did not return any relevant document
    When the prompt is sent to the LLM
    Then the system should instruct the model to respond "I don't have enough information in the knowledge base to answer this question."

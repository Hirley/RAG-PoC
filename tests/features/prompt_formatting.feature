Feature: Prompt formatting

  Scenario: Injecting context and question into the template
    Given the prompt template requires the {context} and {question} tags
    And the search returned 2 text snippets
    When I call the prompt-building function with the question "What is RAG?"
    Then the resulting string should contain the 2 concatenated text snippets
    And the resulting string should contain the original question

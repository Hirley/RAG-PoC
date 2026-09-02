Feature: Search for relevant documents

  Scenario: Question with a match in the database
    Given the database contains a document about "RAG Limitations"
    When I search for the question "When should I not use RAG?"
    Then the system should return a list containing at most 5 documents
    And the document about "RAG Limitations" should be the first result (highest score)

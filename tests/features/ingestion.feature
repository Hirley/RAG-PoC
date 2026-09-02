Feature: Knowledge base ingestion

  Scenario: Successful ingestion of valid documents
    Given I have a list of 100 valid documents in JSON format
    And the index "rag_docs" is ready for use
    When I run the batch indexing function
    Then all 100 documents should be inserted into ElasticSearch
    And the database should confirm the exact record count

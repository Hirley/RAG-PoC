Feature: Automatic index creation

  Scenario: Index does not exist in the database
    Given ElasticSearch is running
    And the index "rag_docs" does not exist
    When the initialization module is executed
    Then the system should successfully create the index "rag_docs"
    And no exception should be raised

  Scenario: Index already exists in the database
    Given ElasticSearch is running
    And the index "rag_docs" already exists
    When the initialization module is executed
    Then the system should detect that the index already exists
    And it should not attempt to recreate it or delete existing data

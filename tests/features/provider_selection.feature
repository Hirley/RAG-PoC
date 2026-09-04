Feature: Choosing an LLM provider

  As an engineer evaluating the pipeline
  I want to answer the same question through different LLM providers
  So that I can compare cost and answer quality without changing code

  Scenario: Answering through the default provider
    Given no provider is configured
    When I ask "When are deploys frozen?" through the pipeline
    Then the request should go to Anthropic
    And the model should be the Anthropic default

  Scenario: Selecting a provider through the environment
    Given the environment selects the "openai" provider
    When I ask "When are deploys frozen?" through the pipeline
    Then the request should go to OpenAI
    And the model should be the OpenAI default

  Scenario: Groq reuses the OpenAI wire format
    Given the environment selects the "groq" provider
    When I ask "When are deploys frozen?" through the pipeline
    Then the request should go to OpenAI
    And the client should point at the Groq endpoint

  Scenario: Overriding the provider for a single call
    Given the environment selects the "openai" provider
    When I ask "When are deploys frozen?" through the "anthropic" provider
    Then the request should go to Anthropic

  Scenario: Each provider carries its own model default
    Given the environment selects the "groq" provider
    And the "GROQ_MODEL" variable names "llama-3.1-8b-instant"
    When I ask "When are deploys frozen?" through the pipeline
    Then the model should be "llama-3.1-8b-instant"

  Scenario: Naming a provider that does not exist
    Given the environment selects the "gemini" provider
    When I ask "When are deploys frozen?" through the pipeline
    Then the run should fail naming the providers that do exist

  Scenario: A truncated answer is refused on every provider
    Given the environment selects the "openai" provider
    And the provider stops generating at the token ceiling
    When I ask "When are deploys frozen?" through the pipeline
    Then the run should fail rather than return the fragment

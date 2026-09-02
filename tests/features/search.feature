Feature: Busca de documentos no índice
  Como um usuário do sistema RAG
  Eu quero buscar documentos relevantes no índice
  Para obter contexto que fundamente a resposta gerada pelo LLM

  Scenario: Busca retorna contexto relevante para uma query válida
    Given o índice de busca contém documentos indexados
    When eu busco pela query "Como funciona o RAG?"
    Then o resultado da busca não deve estar vazio
    And cada resultado deve conter um campo de conteúdo textual

  Scenario: Busca por termo sem correspondência não retorna contexto
    Given o índice de busca contém documentos indexados
    When eu busco pela query "termo completamente inexistente xyzzy"
    Then o resultado da busca deve estar vazio

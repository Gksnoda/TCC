# Chat com Claude 3.7 Sonnet

Um chat inteligente construído com Streamlit que integra Claude 3.7 Sonnet, suporte a upload de arquivos/imagens, busca vetorial com Qdrant e embeddings usando all-MiniLM-L6-v2.

## Funcionalidades

- 💬 **Chat com Claude 3.7 Sonnet**: Conversas naturais com IA avançada
- 📁 **Upload de Arquivos**: Suporte para imagens (PNG, JPG, JPEG) e arquivos de texto (TXT, MD, PY, JSON)
- 🔍 **Busca Vetorial**: Integração com Qdrant para busca semântica
- 🧠 **Embeddings**: Criação automática de embeddings usando all-MiniLM-L6-v2
- 🛠️ **Tools**: Claude pode usar ferramentas para buscar informações relevantes

## Configuração

### 1. Instalar Dependências

```bash
uv add streamlit anthropic qdrant-client sentence-transformers pillow python-dotenv
```

### 2. Configurar Variáveis de Ambiente

Edite o arquivo `config.toml` com suas configurações:

```toml
[api]
anthropic_api_key = "sua_chave_api_anthropic_aqui"

[qdrant]
# Para Qdrant remoto (preferível):
url = "http://seu-servidor-qdrant:6333"
api_key = "sua_api_key_qdrant"

# Para Qdrant local (alternativo):
host = "localhost"
port = 6333

collection_name = "documents"

[embeddings]
model_name = "all-MiniLM-L6-v2"
```

**Nota**: Se você configurar a `url`, ela terá prioridade sobre `host` e `port`.

### 3. Configurar Qdrant

Instale e execute o Qdrant localmente:

```bash
# Usando Docker
docker run -p 6333:6333 qdrant/qdrant

# Ou baixe e execute o binário do Qdrant
```

## Como Usar

### Executar a Aplicação

```bash
uv run streamlit run chat_app.py
```

### Funcionalidades Disponíveis

1. **Chat de Texto**: Digite mensagens normalmente no campo de input
2. **Upload de Arquivos**: Use a sidebar para enviar imagens ou arquivos de texto
3. **Busca Automática**: O sistema automaticamente busca informações relevantes para cada mensagem
4. **Tools do Claude**: O Claude pode usar a ferramenta de busca quando necessário

### Exemplos de Uso

- Envie uma imagem e pergunte sobre ela
- Faça upload de um arquivo de código e peça análise
- Faça perguntas que requerem busca na base de conhecimento
- Use conversas naturais com contexto mantido

## Estrutura do Projeto

```
TCC/
├── chat_app.py          # Aplicação principal
├── config.toml          # Configurações
├── pyproject.toml       # Dependências do projeto
├── main.py             # Arquivo original
└── README.md           # Este arquivo
```

## Tecnologias Utilizadas

- **Streamlit**: Interface web
- **Anthropic Claude 3.7 Sonnet**: Modelo de linguagem
- **Qdrant**: Banco de dados vetorial
- **Sentence Transformers**: Criação de embeddings
- **PIL**: Processamento de imagens
- **UV**: Gerenciamento de dependências

## Requisitos

- Python 3.13+
- Chave API da Anthropic
- Qdrant rodando localmente ou remotamente
- Conexão com internet para download dos modelos

## Notas

- O modelo de embeddings será baixado automaticamente na primeira execução
- As mensagens do usuário são automaticamente salvas no Qdrant para futuras buscas
- O sistema suporta múltiplos tipos de arquivo e formatos de imagem
- Claude pode usar ferramentas para buscar informações quando necessário

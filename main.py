import streamlit as st
import anthropic
import base64
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import json
from typing import List, Dict, Any, Optional
from logger import info, error
import io

import os
import torch
import streamlit
torch.classes.path = []

# Configuração da página
st.set_page_config(
    page_title="Tutor de Sistemas Operacionais ECOS01A - UNIFEI",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_config():
    """Carrega configurações dos secrets do Streamlit"""
    try:
        config = {
            "api": {
                "anthropic_api_key": st.secrets["anthropic_api_key"],
                "ANTHROPIC_MODEL": st.secrets.get("ANTHROPIC_MODEL"),
                "ANTHROPIC_MAX_TOKENS": int(st.secrets.get("ANTHROPIC_MAX_TOKENS"))
            },
            "qdrant": {
                "url": st.secrets["qdrant_url"],
                "api_key": st.secrets["qdrant_api_key"],
                "collection_name": st.secrets.get("qdrant_collection_name")
            },
            "embeddings": {
                "model_name": st.secrets.get("embeddings_model_name")
            }
        }
        return config
    except KeyError as e:
        st.error(f"Variável de ambiente não encontrada: {e}. Configure os secrets no Streamlit Cloud.")
        st.stop()

@st.cache_resource
def initialize_clients():
    """Inicializa os clientes do Anthropic, Qdrant e modelo de embeddings"""
    config = load_config()
    
    # Cliente Anthropic
    anthropic_client = anthropic.Anthropic(
        api_key=config["api"]["anthropic_api_key"]
    )
    
    # Cliente Qdrant - verifica se deve usar URL ou host/port
    qdrant_config = config["qdrant"]
    if "url" in qdrant_config and qdrant_config["url"]:
        # Usa URL e API key se disponível
        qdrant_client = QdrantClient(
            url=qdrant_config["url"],
            api_key=qdrant_config.get("api_key")
        )

    # Modelo de embeddings
    embedding_model = SentenceTransformer(config["embeddings"]["model_name"])
    
    return anthropic_client, qdrant_client, embedding_model, config

def setup_qdrant_collection(qdrant_client, collection_name: str, vector_size: int = 384):
    """Configura a coleção no Qdrant se não existir"""
    try:
        collections = qdrant_client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)
        
        if not collection_exists:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            st.success(f"Coleção '{collection_name}' criada no Qdrant!")
    except Exception as e:
        st.error(f"Erro ao configurar coleção Qdrant: {e}")

def create_embedding(text: str, embedding_model) -> List[float]:
    """Cria embedding do texto usando o modelo all-MiniLM-L6-v2"""
    try:
        info("Criando embedding para o texto: {}", text[:100] + "..." if len(text) > 100 else text)
        embedding = embedding_model.encode(text)
        info("Embedding criado com sucesso. Tamanho do vetor: {}", len(embedding))
        return embedding.tolist()
    except Exception as e:
        error("Erro ao criar embedding: {}", str(e))
        st.error(f"Erro ao criar embedding: {e}")
        return []

def search_qdrant(query_text: str, qdrant_client, embedding_model, config, limit: int = 5) -> List[Dict]:
    """Busca documentos similares no Qdrant"""
    try:
        info("Iniciando busca no Qdrant para: {}", query_text[:100] + "..." if len(query_text) > 100 else query_text)
        query_embedding = create_embedding(query_text, embedding_model)
        
        if not query_embedding:
            error("Embedding vazio para a consulta: {}", query_text)
            return []
        
        info("Executando busca vetorial no Qdrant com limite: {}", limit)
        # Usando query_points em vez de search (deprecado)
        search_results = qdrant_client.query_points(
            collection_name=config["qdrant"]["collection_name"],
            query=query_embedding,
            limit=limit
        )
        
        results = []
        for result in search_results.points:
            results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })
        
        info("Busca no Qdrant concluída. Resultados encontrados: {}", len(results))
        if results:
            info("Melhor resultado (score: {}): {}", results[0]["score"], 
                 results[0]["payload"].get("text", "Sem texto")[:100] + "..." if len(results[0]["payload"].get("text", "")) > 100 else results[0]["payload"].get("text", "Sem texto"))
        
        return results
    except Exception as e:
        error("Erro na busca Qdrant: {}", str(e))
        st.error(f"Erro na busca Qdrant: {e}")
        return []

def encode_image_to_base64(image: Image.Image) -> str:
    """Converte imagem PIL para base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def process_uploaded_file(uploaded_file) -> tuple[str, str, str]:
    """Processa arquivo enviado pelo usuário"""
    file_type = uploaded_file.type
    file_content = ""
    file_info = f"Arquivo: {uploaded_file.name} (Tipo: {file_type})"
    
    if file_type.startswith('image/'):
        image = Image.open(uploaded_file)
        file_content = encode_image_to_base64(image)
        return "image", file_content, file_info

    elif file_type == 'text/plain':
        file_content = str(uploaded_file.read(), "utf-8")
        return "text", file_content, file_info
    else:
        # Para outros tipos de arquivo, lê como texto se possível
        try:
            file_content = str(uploaded_file.read(), "utf-8")
            return "text", file_content, file_info
        except:
            return "unsupported", "", f"Tipo de arquivo não suportado: {file_type}"

def create_claude_message(user_input: str, file_data: Optional[tuple] = None, search_results: List[Dict] = None) -> tuple[str, List[Dict]]:
    """Cria mensagem formatada para o Claude"""
    
    # System prompt especializado
    system_prompt = """Você é um tutor online especializado em ajudar alunos da UNIFEI com a disciplina de Sistemas Operacionais (específicamente a disciplina ECOS01A). Sua função é orientar o aprendizado de forma pedagógica e progressiva.

## Diretrizes de Comportamento:

### Comunicação:
- Responda sempre em português brasileiro
- Use linguagem clara, didática e acessível
- Seja paciente e encorajador
- Mantenha um tom amigável e profissional

### Metodologia de Ensino:
- Quando um aluno pedir ajuda com exercícios, PRIMEIRO incentive-o a tentar resolver sozinho
- Forneça dicas iniciais e direcionamentos para começar a resolução
- Crie exemplos práticos e situações do cotidiano para ilustrar conceitos
- Use analogias simples quando necessário para facilitar o entendimento

### Resolução de Exercícios:
- Se o aluno demonstrar dificuldade real, você pode guiá-lo através do processo completo
- Explique cada passo da resolução de forma detalhada
- Conduza o aluno através de todo o raciocínio até a etapa final
- **IMPORTANTE**: Você NUNCA deve fornecer a resposta final definitiva
- Pare sempre no penúltimo passo, deixando que o aluno complete a solução

### Recursos Disponíveis:
- Você tem acesso a uma ferramenta de pesquisa no Qdrant com documentos da disciplina de Sistemas Operacionais da UNIFEI
- Use esses recursos para fundamentar suas explicações quando necessário
- Referencie o material oficial sempre que relevante

### Objetivo:
Desenvolver o raciocínio crítico e a autonomia do aluno, garantindo que ele compreenda os conceitos e processos, não apenas obtenha respostas prontas."""

    content = []
    
    # Adiciona texto do usuário
    if user_input:
        content.append({
            "type": "text",
            "text": user_input
        })
    
    # Adiciona resultados da busca se disponíveis
    if search_results:
        search_text = "\n\nResultados da busca no conhecimento:\n"
        for i, result in enumerate(search_results, 1):
            search_text += f"{i}. (Score: {result['score']:.3f}) {result['payload'].get('text', 'Sem texto')}\n"
        
        content.append({
            "type": "text",
            "text": search_text
        })
    
    # Adiciona arquivo se fornecido
    if file_data:
        file_type, file_content, file_info = file_data
        
        if file_type == "image":
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": file_content
                }
            })
            content.append({
                "type": "text",
                "text": f"\n{file_info}"
            })

        elif file_type == "text":
            content.append({
                "type": "text",
                "text": f"\n{file_info}\nConteúdo do arquivo:\n{file_content}"
            })
        elif file_type == "error":
            content.append({
                "type": "text",
                "text": f"\n❌ {file_info}"
            })
    
    # Retorna o system prompt e as mensagens separadamente
    messages = [{"role": "user", "content": content}]
    
    return system_prompt, messages

def call_claude_with_tools(anthropic_client, messages: List[Dict], search_function, config, system_prompt: str) -> str:
    """Chama Claude com ferramentas disponíveis e streaming"""
    tools = [
        {
            "name": "search_knowledge",
            "description": "Busca informações relevantes na base de conhecimento usando busca vetorial",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Texto da consulta para buscar informações relevantes"
                    }
                },
                "required": ["query"]
            }
        }
    ]
    
    try:
        # Primeira chamada para verificar se Claude quer usar ferramentas
        response = anthropic_client.messages.create(
            model=config["api"]["ANTHROPIC_MODEL"],
            max_tokens=config["api"]["ANTHROPIC_MAX_TOKENS"],
            system=system_prompt,
            tools=tools,
            messages=messages
        )
        
        # Processa resposta e possíveis chamadas de ferramentas
        if response.stop_reason == "tool_use":
            # Claude quer usar uma ferramenta
            tool_calls = [block for block in response.content if block.type == "tool_use"]
            
            # Adiciona resposta do Claude às mensagens
            messages.append({"role": "assistant", "content": response.content})
            
            # Processa cada chamada de ferramenta
            tool_results = []
            for tool_call in tool_calls:
                if tool_call.name == "search_knowledge":
                    query = tool_call.input["query"]
                    info("Claude solicitou busca por: {}", query)
                    search_results = search_function(query)
                    
                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": json.dumps(search_results, ensure_ascii=False)
                    }
                    tool_results.append(tool_result)
            
            # Adiciona resultados das ferramentas
            messages.append({"role": "user", "content": tool_results})
            
            # Retorna um objeto que pode ser usado com st.write_stream
            def stream_generator():
                with anthropic_client.messages.stream(
                    model=config["api"]["ANTHROPIC_MODEL"],
                    max_tokens=config["api"]["ANTHROPIC_MAX_TOKENS"],
                    system=system_prompt,
                    messages=messages
                ) as stream:
                    for text_chunk in stream.text_stream:
                        yield text_chunk
            return stream_generator()
        else:
            # Resposta direta, criar stream
            def stream_generator():
                with anthropic_client.messages.stream(
                    model=config["api"]["ANTHROPIC_MODEL"],
                    max_tokens=config["api"]["ANTHROPIC_MAX_TOKENS"],
                    system=system_prompt,
                    messages=messages
                ) as stream:
                    for text_chunk in stream.text_stream:
                        yield text_chunk
            return stream_generator()
            
    except Exception as e:
        error("Erro ao chamar Claude: {}", str(e))
        return f"Erro ao chamar Claude: {e}"

def main():
    st.title("🎓 Tutor de Sistemas Operacionais ECOS01A - UNIFEI")
    st.markdown("**Assistente pedagógico especializado em Sistemas Operacionais** • Suporte a textos, imagens e busca no conhecimento")
    
    # Inicializa clientes
    try:
        anthropic_client, qdrant_client, embedding_model, config = initialize_clients()
        setup_qdrant_collection(qdrant_client, config["qdrant"]["collection_name"])
    except Exception as e:
        st.error(f"Erro na inicialização: {e}")
        st.stop()
    
    # Inicializa histórico de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Botão para limpar chat
        if st.button("🗑️ Limpar Chat"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📁 Suporte a Arquivos")
        st.info("✅ **Imagens**: JPEG, PNG, GIF, WebP\n")
        st.info("Use o botão de anexo no chat para enviar arquivos!")
        
        st.markdown("---")
        st.markdown("### 🔍 Busca Vetorial")
        st.info("O sistema automaticamente busca informações relevantes na base de conhecimento para cada mensagem.")
    
    # Exibe histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input do usuário com suporte a arquivos
    message = st.chat_input(
        "Digite sua mensagem...",
        accept_file=True,
        file_type=["jpg", "jpeg", "png", "gif", "webp", "txt"]
    )
    
    if message:
        # Extrair corretamente o texto e arquivos do ChatInputValue
        if hasattr(message, 'text'):
            user_input = message.text or ""
            uploaded_files = getattr(message, 'files', []) or []
        else:
            # Fallback para compatibilidade
            user_input = message if isinstance(message, str) else str(message)
            uploaded_files = []
        
        # Log do input do usuário
        info("Input do usuário recebido: {}", user_input[:200] + "..." if len(user_input) > 200 else user_input)
        if uploaded_files:
            info("Arquivos enviados: {}", [f.name for f in uploaded_files])
        
        # Adiciona mensagem do usuário ao histórico
        display_text = user_input
        if uploaded_files:
            file_names = ", ".join([f.name for f in uploaded_files])
            display_text += f"\n\n📎 **Arquivos anexados:** {file_names}"
        
        st.session_state.messages.append({"role": "user", "content": display_text})
        
        with st.chat_message("user"):
            st.markdown(display_text)
        
        # Cria embedding da mensagem do usuário
        with st.spinner("Criando embedding da mensagem..."):
            user_embedding = create_embedding(user_input, embedding_model)
        
        # Processa arquivos se enviados
        file_data = None
        if uploaded_files:
            # Processa apenas o primeiro arquivo por enquanto
            uploaded_file = uploaded_files[0]
            with st.spinner(f"Processando arquivo {uploaded_file.name}..."):
                file_data = process_uploaded_file(uploaded_file)
                if file_data[0] == "error":
                    st.error(file_data[2])
        
        # Busca no Qdrant
        with st.spinner("Buscando informações relevantes..."):
            search_results = search_qdrant(user_input, qdrant_client, embedding_model, config)
        
        # Cria função de busca para as ferramentas do Claude
        def search_function(query: str) -> List[Dict]:
            return search_qdrant(query, qdrant_client, embedding_model, config)
        
        # Prepara mensagens para o Claude
        system_prompt, claude_messages = create_claude_message(user_input, file_data, search_results)
        
        # Chama Claude com streaming
        with st.chat_message("assistant"):
            with st.spinner("Claude está pensando..."):
                stream_or_error = call_claude_with_tools(anthropic_client, claude_messages, search_function, config, system_prompt)
            
            if isinstance(stream_or_error, str):
                # Erro
                response = stream_or_error
                st.markdown(response)
            else:
                # Usando o método nativo st.write_stream com tratamento de erro
                try:
                    # Log para debug
                    info("Iniciando streaming com st.write_stream")
                    
                    # Utilizamos o write_stream do Streamlit, que já tem tratamento interno
                    response = st.write_stream(stream_or_error)
                    
                    # Log após o streaming
                    info("Streaming concluído com sucesso")
                    
                    # Se a resposta estiver vazia, exibimos mensagem
                    if not response:
                        fallback_response = "Resposta recebida, mas sem conteúdo textual."
                        st.markdown(fallback_response)
                        response = fallback_response
                        
                except Exception as e:
                    error(f"Erro ao processar stream: {e}")
                    error_msg = f"Erro ao processar a resposta: {e}"
                    st.error(error_msg)
                    response = error_msg
        
        # Adiciona resposta do assistente ao histórico
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Salva embedding da mensagem do usuário no Qdrant (opcional)
        if user_embedding:
            try:
                point_id = str(uuid.uuid4())
                qdrant_client.upsert(
                    collection_name=config["qdrant"]["collection_name"],
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=user_embedding,
                            payload={
                                "text": user_input,
                                "timestamp": str(st.session_state.get("timestamp", "unknown")),
                                "type": "user_message"
                            }
                        )
                    ]
                )
            except Exception as e:
                st.error(f"Erro ao salvar no Qdrant: {e}")

if __name__ == "__main__":
    main() 
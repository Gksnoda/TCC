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

# Constantes para cálculo de custos
COST_INPUT_PER_MILLION = 3.0  # $3 por 1 milhão de tokens de input
COST_OUTPUT_PER_MILLION = 15.0  # $15 por 1 milhão de tokens de output

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
            check_compatibility=False,
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
            info("=== RESULTADOS DA BUSCA ===")
            for i, result in enumerate(results, 1):
                # Buscar texto no campo correto
                result_text = result["payload"].get("text_content", "")
                if not result_text:
                    result_text = result["payload"].get("text", "Sem texto")
                
                # Mostrar apenas o comecinho do texto (primeiras 30 palavras para não ficar muito longo)
                words = result_text.split()
                if len(words) > 30:
                    preview_text = " ".join(words[:30]) + "..."
                else:
                    preview_text = result_text
                
                info("Resultado {} (score: {:.6f}): {}", i, result["score"], preview_text)
                info("Documento {}: {}", i, result["payload"].get("document_name", "Documento não identificado"))
                if i < len(results):  # Adiciona separador se não for o último
                    info("---")
        
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

def create_claude_message(user_input: str, file_data: Optional[tuple] = None) -> tuple[str, List[Dict]]:
    """Cria mensagem formatada para o Claude"""
    
    # System prompt especializado
    system_prompt = """Você é um tutor online especializado em ajudar alunos da UNIFEI com a disciplina de Sistemas Operacionais (específicamente a disciplina ECOS01A). Sua função é orientar o aprendizado de forma pedagógica e progressiva.

## Diretrizes de Comportamento:

### Comunicação:
    - Responda sempre em português brasileiro
    - Use linguagem clara, didática e acessível
    - Seja paciente e encorajador
    - Mantenha um tom amigável e profissional
    - Sempre que for responder o aluno com a resposta da ferramenta, cite qual pagina voce tirou a resposta

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
    
    # Adiciona texto do usuário
    if user_input:
        content.append({
            "type": "text",
            "text": user_input
        })
    
    # Retorna o system prompt e as mensagens separadamente
    messages = [{"role": "user", "content": content}]
    
    return system_prompt, messages

def call_claude_with_tools(anthropic_client, messages: List[Dict], search_function, config, system_prompt: str):
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
    
    # Classe para armazenar tokens capturados durante streaming
    class TokenCounter:
        def __init__(self):
            self.input_tokens = 0
            self.output_tokens = 0
    
    token_counter = TokenCounter()
    
    def main_stream_generator():
        try:
            # Primeira chamada COM STREAMING REAL
            collected_text = ""
            with anthropic_client.messages.stream(
                model=config["api"]["ANTHROPIC_MODEL"],
                max_tokens=config["api"]["ANTHROPIC_MAX_TOKENS"],
                system=system_prompt,
                tools=tools,
                messages=messages,
            ) as stream:
                # STREAMING EM TEMPO REAL DA PRIMEIRA CHAMADA
                for text in stream.text_stream:
                    collected_text += text
                    yield text  # Mostra em tempo real
                
                # Captura response após consumir stream
                response = stream.get_final_message()
            
            # Sempre conta tokens da primeira chamada
            token_counter.input_tokens += response.usage.input_tokens
            token_counter.output_tokens += response.usage.output_tokens
            info("Tokens primeira chamada - Input: {}, Output: {}", response.usage.input_tokens, response.usage.output_tokens)

            # Verifica se precisa de ferramentas
            if response.stop_reason == "tool_use":
                info("Claude solicitou uso de ferramenta")
                
                # Claude quer usar uma ferramenta
                tool_calls = [block for block in response.content if block.type == "tool_use"]
                
                # Processa ferramentas
                messages.append({"role": "assistant", "content": response.content})
                
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
                
                messages.append({"role": "user", "content": tool_results})
                
                # Mostra indicador de busca e segunda chamada COM STREAMING
                yield "\n\n🔍 *Buscando informações na base de conhecimento...*\n\n"
                
                # Segunda chamada com streaming
                with anthropic_client.messages.stream(
                    model=config["api"]["ANTHROPIC_MODEL"],
                    max_tokens=config["api"]["ANTHROPIC_MAX_TOKENS"],
                    system=system_prompt,
                    messages=messages
                ) as stream:
                    for text_chunk in stream.text_stream:
                        yield text_chunk
                    
                    # Captura tokens da segunda chamada também
                    final_message = stream.get_final_message()
                    token_counter.input_tokens += final_message.usage.input_tokens
                    token_counter.output_tokens += final_message.usage.output_tokens
                    info("Tokens segunda chamada - Input: {}, Output: {}", final_message.usage.input_tokens, final_message.usage.output_tokens)
            
            # Log final dos tokens
            info("=== RESUMO TOKENS DESTA INTERAÇÃO ===")
            info("TOTAL - Input: {}, Output: {}", token_counter.input_tokens, token_counter.output_tokens)
                
        except Exception as e:
            error("Erro ao chamar Claude: {}", str(e))
            yield f"Erro ao chamar Claude: {e}"
    
    return main_stream_generator, token_counter

def initialize_session_state():
    """Inicializa variáveis de estado da sessão"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_input_tokens" not in st.session_state:
        st.session_state.total_input_tokens = 0
    if "total_output_tokens" not in st.session_state:
        st.session_state.total_output_tokens = 0
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "last_call_cost" not in st.session_state:
        st.session_state.last_call_cost = 0.0

def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calcula o custo baseado nos tokens de input e output"""
    input_cost = (input_tokens / 1_000_000) * COST_INPUT_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * COST_OUTPUT_PER_MILLION
    return input_cost + output_cost

def update_token_usage(input_tokens: int, output_tokens: int):
    """Atualiza o uso de tokens e custos na sessão"""
    st.session_state.total_input_tokens += input_tokens
    st.session_state.total_output_tokens += output_tokens
    
    # Calcula custo DESTA chamada específica
    call_cost = calculate_cost(input_tokens, output_tokens)
    st.session_state.last_call_cost = call_cost
    
    # Adiciona ao custo TOTAL da sessão
    st.session_state.total_cost += call_cost
    
    info("Tokens usados - Input: {}, Output: {}, Custo desta chamada: ${:.6f}, Custo total sessão: ${:.6f}", 
         input_tokens, output_tokens, call_cost, st.session_state.total_cost)

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
    initialize_session_state()
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Botão para limpar chat
        if st.button("🗑️ Limpar Chat"):
            st.session_state.messages = []
            st.session_state.total_input_tokens = 0
            st.session_state.total_output_tokens = 0
            st.session_state.total_cost = 0.0
            st.session_state.last_call_cost = 0.0
            st.rerun()
        
        st.markdown("---")
        
        # Menu de tokens e custos
        with st.expander("💰 Uso de Tokens e Custos", expanded=False):
            st.markdown("### 📊 Estatísticas da Sessão")
            
            # Tokens
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📥 Input Tokens", f"{st.session_state.total_input_tokens:,}")
                st.metric("📤 Output Tokens", f"{st.session_state.total_output_tokens:,}")
            
            with col2:
                total_tokens = st.session_state.total_input_tokens + st.session_state.total_output_tokens
                st.metric("🎯 Total Tokens", f"{total_tokens:,}")
            
            # Custos
            st.markdown("---")
            col3, col4 = st.columns(2)
            with col3:
                st.metric("💸 Última Chamada", f"${st.session_state.last_call_cost:.6f}")
            
            with col4:
                st.metric("💰 Custo Total Sessão", f"${st.session_state.total_cost:.6f}")
            
            st.markdown("---")
            st.markdown("**💡 Preços por 1M de tokens:**")
            st.markdown(f"• Input: ${COST_INPUT_PER_MILLION:.2f}")
            st.markdown(f"• Output: ${COST_OUTPUT_PER_MILLION:.2f}")
            
            st.markdown("---")
            st.markdown("**💸 Última Chamada**: Custo apenas da mensagem atual")
            st.markdown("**💰 Custo Total**: Soma de todas as chamadas da sessão")
        
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
        
        # Processa arquivos se enviados
        file_data = None
        if uploaded_files:
            # Processa apenas o primeiro arquivo por enquanto
            uploaded_file = uploaded_files[0]
            with st.spinner(f"Processando arquivo {uploaded_file.name}..."):
                file_data = process_uploaded_file(uploaded_file)
                if file_data[0] == "error":
                    st.error(file_data[2])
        
        # Cria função de busca para as ferramentas do Claude
        def search_function(query: str) -> List[Dict]:
            return search_qdrant(query, qdrant_client, embedding_model, config, limit=3)
        
        # Prepara mensagens para o Claude - INCLUI TODO O HISTÓRICO
        system_prompt, claude_messages = create_claude_message(user_input, file_data)
        
        # Adiciona todo o histórico da conversa às mensagens do Claude
        full_messages = []
        for msg in st.session_state.messages[:-1]:  # Exclui a última (que acabamos de adicionar)
            if msg["role"] == "user":
                full_messages.append({"role": "user", "content": msg["content"]})
            else:
                full_messages.append({"role": "assistant", "content": msg["content"]})
        
        # Adiciona a mensagem atual do usuário
        full_messages.extend(claude_messages)
        
        # Limpa tokens antes de processar nova mensagem
        info("Limpando contadores de tokens para nova mensagem")
        # Tokens são automaticamente limpos porque TokenCounter é criado novo a cada chamada
        
        # Mostra "Tutor está pensando..." APÓS o usuário enviar a mensagem
        thinking_placeholder = st.empty()
        with thinking_placeholder:
            st.info("🤔 Tutor está pensando...")
        
        # Chama Claude
        with st.chat_message("assistant"):
            response = ""  # Inicializa response
            # Remove o "pensando" quando começar a responder
            thinking_placeholder.empty()
            result = call_claude_with_tools(anthropic_client, full_messages, search_function, config, system_prompt)
            
            if isinstance(result, tuple) and len(result) == 2:
                stream_generator, token_counter = result
                
                if callable(stream_generator):
                    # Usa streaming em tempo real
                    response = st.write_stream(stream_generator)
                    
                    # Atualiza o uso de tokens e custos APÓS o streaming completar
                    update_token_usage(token_counter.input_tokens, token_counter.output_tokens)
                    
                    # Adiciona resposta do assistente ao histórico ANTES do rerun
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Força atualização da UI para mostrar novos tokens
                    st.rerun()
                else:
                    response = "Erro: stream_generator não é chamável"
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                response = "Erro na chamada do Claude"
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 
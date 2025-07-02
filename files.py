import os

def processar_arquivos_individualmente(caminho_da_pasta='parsed'):
    """
    Para cada arquivo .md em uma pasta, lê seu conteúdo,
    prepara o texto para ser seguro em um JSON (escapa barras e aspas),
    transforma-o em uma única linha e o salva de volta no arquivo original.

    Args:
        caminho_da_pasta (str): O caminho para a pasta contendo os arquivos .md.
                                O padrão é 'parsed'.
    """
    # Verifica se a pasta existe
    if not os.path.isdir(caminho_da_pasta):
        print(f"Erro: A pasta '{caminho_da_pasta}' não foi encontrada.")
        return

    # Lista todos os arquivos na pasta que terminam com .md
    nomes_dos_arquivos = [f for f in os.listdir(caminho_da_pasta) if f.endswith('.md')]

    if not nomes_dos_arquivos:
        print(f"Nenhum arquivo .md encontrado na pasta '{caminho_da_pasta}'.")
        return

    print(f"Encontrados {len(nomes_dos_arquivos)} arquivos .md. Processando...")

    # Itera sobre cada arquivo .md encontrado
    for nome_do_arquivo in nomes_dos_arquivos:
        caminho_completo = os.path.join(caminho_da_pasta, nome_do_arquivo)
        try:
            # --- Etapa 1: Ler o conteúdo original do arquivo ---
            with open(caminho_completo, 'r', encoding='utf-8') as arquivo:
                conteudo_original = arquivo.read()

            # --- Etapa 2: Modificar o conteúdo para ser seguro em JSON ---
            # A ORDEM É IMPORTANTE:
            # 1. Escapa as barras invertidas (substitui \ por \\)
            # 2. Escapa as aspas duplas (substitui " por \")
            # 3. Substitui as quebras de linha por um espaço.
            conteudo_modificado = conteudo_original.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')

            # --- Etapa 3: Salvar o novo conteúdo no mesmo arquivo ---
            # Abre o arquivo em modo de escrita ('w'), o que apaga o conteúdo anterior
            with open(caminho_completo, 'w', encoding='utf-8') as arquivo:
                arquivo.write(conteudo_modificado)

            print(f"O arquivo '{nome_do_arquivo}' foi processado e salvo com sucesso.")

        except Exception as e:
            print(f"Ocorreu um erro ao processar o arquivo {nome_do_arquivo}: {e}")

# --- Execução do Script ---

# Define o nome da pasta onde estão os seus arquivos .md
pasta_alvo = 'parsed'

# Chama a função para processar os arquivos
processar_arquivos_individualmente(pasta_alvo)

print("\nProcesso concluído!")

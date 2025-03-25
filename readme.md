## Passos para Configuração

1. **Instalar pyenv**

2. **Altrar para versão compatível do python através do pyenv**

   ```sh
   pyenv install 3.10.16
   pyenv local 3.10.16
   ```

3. **Criar um ambiente virtual**

   ```sh
   python -m venv env
   ```

2. **Ativar o ambiente virtual**

   ```sh
   source env/bin/activate
   ```

3. **Instalar as dependências do projeto**

   ```sh
   python install_requirements.py
   ```

Após seguir esses passos, o ambiente estará pronto para rodar o projeto. Certifique-se de sempre ativar o ambiente virtual antes de executar qualquer comando relacionado ao projeto.


# Imagen base optimizada para Lambda con Python
FROM public.ecr.aws/lambda/python:3.12

# Establecer directorio de trabajo dentro del contenedor
WORKDIR /var/task

# Copiar archivos del proyecto al contenedor
COPY . ${LAMBDA_TASK_ROOT}/

# Instalar dependencias
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Comando para ejecutar Lambda
ENTRYPOINT ["/var/runtime/bootstrap"]
CMD ["app.lambda_handler"]

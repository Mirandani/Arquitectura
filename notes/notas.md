

AWS Copiar archivos desde mi local a un bucket S3
===========================================
Para copiar archivos desde tu máquina local a un bucket de Amazon S3, puedes utilizar la herramienta de línea de comandos de AWS (AWS CLI). A continuación, se describen los pasos para hacerlo:
1. **Instalar AWS CLI**: Si aún no tienes AWS CLI instalado, puedes descargarlo e instalarlo siguiendo las instrucciones oficiales en [AWS CLI Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html).
2. **Configurar AWS CLI**: Después de instalar AWS CLI, necesitas configurarlo con tus credenciales de AWS. Ejecuta el siguiente comando y proporciona tu Access Key ID, Secret Access Key, región predeterminada y formato de salida preferido:
   ```
   aws configure
   ```
3. **Copiar archivos al bucket S3**: Utiliza el comando `aws s3 cp` para copiar archivos desde tu máquina local al bucket S3. La sintaxis básica es

    ```
    aws s3 cp <ruta_local> s3://<nombre_del_bucket>/<ruta_destino>
    ```
    Por ejemplo, para copiar un archivo llamado `archivo.txt` desde tu escritorio a un bucket llamado `mi-bucket` en una carpeta llamada `carpeta-destino`, usarías el siguiente comando:
    ```
    aws s3 cp ~/Escritorio/archivo.txt s3://mi-bucket/carpeta-destino/
    ```
4. ** copiar desde aws s3 a local**
   Para copiar archivos desde un bucket S3 a tu máquina local, puedes usar el mismo comando `aws s3 cp`, pero invirtiendo las rutas. La sintaxis sería:
   ```
   aws s3 cp s3://<nombre_del_bucket>/<ruta_origen> <ruta_local>
   ```
   Por ejemplo, para copiar un archivo llamado `archivo.txt` desde el bucket `mi-bucket` en la carpeta `carpeta-origen` a tu escritorio, usarías el siguiente comando:
   ```
   aws s3 cp s3://mi-bucket/carpeta-origen/archivo.txt ~/Escritorio/
   ```

  aws cp /Users/mb95676/ITAM/Arquitectura/notes/notas.md s3://demo-9564-6312-3241/
  aws s3 cp /Users/mb95676/ITAM/Arquitectura/notes/notas.md s3://demo-9564-6312-3241/


   
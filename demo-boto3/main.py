import boto3

# los clientes seon los servicios de aws
# cada vez que haga uso de s3, necesito crear un cliente
s3 = boto3.client('s3')

#listar los buckets
response = s3.list_buckets()

print("Buckets existentes:")
for bucket in response['Buckets']:
    print(f"  {bucket['Name']}")
    
    

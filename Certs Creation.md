# Certificate creation for Secure Kafka

## Generate CA
```
openssl req -new -x509 -keyout ca-key.pem -out ca-cert.pem -days 365 \
    -nodes -subj "/CN=Kafka-CA"
```
openssl req -new -x509 -keyout ca-key.pem -out ca-cert.pem -days 365 -nodes -subj "/CN=Kafka-CA"

## Kafka Side

### Kakfa keystore
```
keytool -genkey -alias kafka \
    -keystore kafka.keystore.jks \
    -keyalg RSA -storepass password -keypass password \
    -dname "CN=localhost"
```
keytool -genkey -alias kafka -keystore kafka.keystore.jks -keyalg RSA -storepass password -keypass password -dname "CN=localhost"

### Kafka CSR
```
keytool -keystore kafka.keystore.jks -alias kafka -certreq \
    -file kafka.csr -storepass password
```
keytool -keystore kafka.keystore.jks -alias kafka -certreq -file kafka.csr -storepass password

### Sign Kafka Cerificate
```
openssl x509 -req -CA ca-cert.pem -CAkey ca-key.pem -in kafka.csr \
    -out kafka-cert-signed.pem -days 365 -CAcreateserial
```
openssl x509 -req -CA ca-cert.pem -CAkey ca-key.pem -in kafka.csr -out kafka-cert-signed.pem -days 365 -CAcreateserial

### Import CA in Kafka keystore
```
keytool -keystore kafka.keystore.jks -alias CARoot \
    -import -file ca-cert.pem -storepass password -noprompt
```
keytool -keystore kafka.keystore.jks -alias CARoot -import -file ca-cert.pem -storepass password -noprompt

### Import Kafka Signed Certificate
```
keytool -keystore kafka.keystore.jks -alias kafka \
    -import -file kafka-cert-signed.pem -storepass password -noprompt
```
keytool -keystore kafka.keystore.jks -alias kafka -import -file kafka-cert-signed.pem -storepass password -noprompt

### Create Kafka Truststore
```
keytool -keystore kafka.truststore.jks -alias CARoot \
    -import -file ca-cert.pem -storepass password -noprompt
```
keytool -keystore kafka.truststore.jks -alias CARoot -import -file ca-cert.pem -storepass password -noprompt

## Client side

### Generate Client key
```
openssl genrsa -out client-key.pem 2048
```
openssl genrsa -out client-key.pem 2048

### Client CSR
```
openssl req -new -key client-key.pem -out client.csr \
    -subj "/CN=Kafka-Client"
```
openssl req -new -key client-key.pem -out client.csr -subj "/CN=Kafka-Client"

### Sign Client certificate
```
openssl x509 -req -CA ca-cert.pem -CAkey ca-key.pem -in client.csr \
    -out client-cert.pem -days 365 -CAcreateserial
```
openssl x509 -req -CA ca-cert.pem -CAkey ca-key.pem -in client.csr -out client-cert.pem -days 365 -CAcreateserial
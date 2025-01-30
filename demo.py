from kafka.admin import KafkaAdminClient, NewTopic

admin_client = KafkaAdminClient(
    bootstrap_servers="localhost:29092", 
    client_id='test'
)

topic_list = []
# topic_list.append(NewTopic(name="global-models", num_partitions=1, replication_factor=1))
# topic_list.append(NewTopic(name="client-logs", num_partitions=1, replication_factor=1))
topic_list.append(NewTopic(name="local-models", num_partitions=1, replication_factor=1))
admin_client.create_topics(new_topics=topic_list, validate_only=False)

print(admin_client.list_topics())

# admin_client.delete_topics(topics=["example_topic"])
# print(admin_client.list_topics())
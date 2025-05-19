package com.example.podspawner.service;

import org.springframework.stereotype.Service;

import io.kubernetes.client.openapi.ApiClient;
import io.kubernetes.client.openapi.Configuration;
import io.kubernetes.client.openapi.apis.AppsV1Api;
import io.kubernetes.client.openapi.models.V1Container;
import io.kubernetes.client.openapi.models.V1ObjectMeta;
import io.kubernetes.client.openapi.models.V1PodSpec;
import io.kubernetes.client.openapi.models.V1PodTemplateSpec;
import io.kubernetes.client.openapi.models.V1DeploymentBuilder;
import io.kubernetes.client.openapi.models.V1DeploymentSpec;
import io.kubernetes.client.openapi.models.V1LabelSelector;
import io.kubernetes.client.util.Config;
import java.util.Collections;

@Service
public class KubernetesService {

    public void createPod(String podName) throws Exception {

        // Inspired by
        // https://github.com/kubernetes-client/java/blob/master/examples/examples-release-20/src/main/java/io/kubernetes/client/examples/DeployRolloutRestartExample.java
        ApiClient client = Config.defaultClient();
        Configuration.setDefaultApiClient(client);
        AppsV1Api appsV1Api = new AppsV1Api(client);

        String namespace = "default";
        String deploymentName = podName + "-deployment";
        String imageName = "deglazed/test-frontend:latest";

        V1DeploymentBuilder deploymentBuilder = new V1DeploymentBuilder()
                .withApiVersion("apps/v1")
                .withKind("Deployment")
                .withMetadata(new V1ObjectMeta().name(deploymentName).namespace(namespace))
                .withSpec(new V1DeploymentSpec()
                        .replicas(1)
                        .selector(new V1LabelSelector().putMatchLabelsItem("name", deploymentName))
                        .template(new V1PodTemplateSpec()
                                .metadata(new V1ObjectMeta().putLabelsItem("name", deploymentName))
                                .spec(new V1PodSpec()
                                        .containers(Collections.singletonList(
                                                new V1Container().name(deploymentName).image(imageName))))));

        appsV1Api.createNamespacedDeployment(namespace, deploymentBuilder.build()).execute();

    }
}

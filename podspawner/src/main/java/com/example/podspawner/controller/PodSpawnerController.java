package com.example.podspawner.controller;

import com.example.podspawner.service.KubernetesService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class PodSpawnerController {
    
    @Autowired
    private KubernetesService kubernetesService;

    @PostMapping("/spawn-pod")
    public String spawnPod(@RequestParam String podName) {
        try {
            kubernetesService.createPod(podName);
            return  "Pod " + podName + " created successfully.";
        } catch (Exception e) {
            return "Error creating pod " + podName + ": " + e.getMessage();
        }
    }
}

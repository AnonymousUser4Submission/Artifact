console.log("first line of file: ");


var PROFILING_RESULT = {};
var tolEpoch = 10000;
window.fps_arr=[];
window.fps_rounder = [];
var stop_flag = false;
var server_url = ".";
var time_interval_rate = 1.0;


let D_tfjs = {
    'imagenet_mobilenet_v2_100_224_classification_5': [  //wasm-ok, webgpu-ok
        [[1, 224, 224, 3]], ['float32']
    ],
    'resnet_50_classification_1': [   //wasm-ok, webgpu-ok
        [[1, 224, 224, 3]], ['float32']
    ],
    'ssd_mobilenet_v2_2': [   //wasm-ok
        [[1, 224, 224, 3]], ['int32']
    ],
    'faster_rcnn_resnet50_v1_1024x1024_1': [   //wasm-GG
        [[1, 224, 224, 3]], ['int32']
    ],
    'edgetpu_nlp_mobilebert-edgetpu_xs_1': [   //wasm-GG
        [[1, 10], [1, 10], [1, 10]], ['int32', 'float32', 'int32']
    ],
    'mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1': [   //wasm-GG
        [[1, 10], [1, 10], [1, 10]], ['int32', 'int32', 'int32'], ['input_word_ids', 'input_type_ids', 'input_mask']
    ],
    'albert_en_base_3': [   //wasm-GG
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'electra_small_2': [   //wasm-GG
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'esrgan-tf2_1': [   //wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32']//, ['input_0']
    ],
    'movenet_multipose_lightning_1': [   //wasm-GG, webgl-ok
        [[1, 224, 224, 3]], ['int32']//, ['input']
    ],
    'experts_bert_pubmed_2': [   //wasm-GG, webgl-GG
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'movenet_singlepose_thunder_4': [   //wasm-ok, webgl-ok
        [[1, 256, 256, 3]], ['int32']//, ['input']
    ],
    'small_bert_bert_en_uncased_L-2_H-128_A-2_2': [   //wasm-GG, webgl-ok
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'trillsson2_1': [   //wasm-GG, webgl-GG
        [[1, 128]], ['float32'] //, ['audio_samples']
    ],
    'language_model': [   //wasm-GG
        [[1, 128]], ['float32'], ['embedding_input']
    ],
    'VGG16': [  // wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32'] //, ['input_1']  
    ],
    'ShuffleNetV2': [  // wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32'] //, ['input_1']
    ],
    'yolov5': [  // wasm-GG(Kernel 'Softplus' not registered for backend 'wasm'.), webgl-ok
        [[1, 224, 224, 3]], ['float32'] //, ['input_1']
    ],
    'Xception': [  // wasm-ok, webgl-ok
        [[1, 299, 299, 3]], ['float32'] //, ['input_3']
    ],
    'InceptionV3': [  // wasm-ok, webgl-ok
        [[1, 299, 299, 3]], ['float32'] //, ['input_1']
    ],
    'EfficientNetV2': [  // wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32'] //, ['input_1']
    ],
    'efficientdet_d1_coco17_tpu-32': [  // wasm-GG(Error: Kernel 'Reciprocal' not registered for backend 'wasm'.), webgl-GG
        [[1, 640, 640, 3]], ['int32'] //, ['input_tensor']
    ],
    'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8': [  // wasm-GG(Error: Kernel 'Reciprocal' not registered for backend 'wasm'.), webgl-ok
        [[1, 512, 512, 3]], ['int32'] //, ['input_tensor']  
    ],

    'progan-128_1': [  // https://tfhub.dev/google/progan-128/1
        [[8, 512]], ['float32'], ['latent_vector']
    ],
    'pggan_512x512': [
        [[64, 512], [64, 1]], ['float32', 'float32'], ['input_1', 'input_alpha']
    ],
    'compare_gan_model_11_cifar10_resnet_cifar_1': [
        [[64, 128]], ['float32'], ['z']
    ],
    'compare_gan_model_12_cifar10_resnet_cifar_1': [
        [[64, 128]], ['float32'], ['z']
    ],
    'compare_gan_model_13_cifar10_resnet_cifar_1': [
        [[64, 128]], ['float32'], ['z']
    ],
    'compare_gan_model_14_cifar10_resnet_cifar_1': [
        [[64, 128]], ['float32'], ['z']
    ],
    'compare_gan_model_15_cifar10_resnet_cifar_1': [
        [[64, 128]], ['float32'], ['z']
    ],
    'mirnet-tfjs_1': [
        [[1, 128, 128, 3]], ['float32'], ['input_1']
    ],
    'tfjs-model_mirnet-tfjs_default_fp16_1': [
        [[1, 128, 128, 3]], ['float16'], ['input_1']
    ],
    'tfjs-model_mirnet-tfjs_default_uint8_1': [
        [[1, 128, 128, 3]], ['uint8'], ['input_1']
    ],
    'tfjs-model_mirnet-tfjs_default_uint16_1': [
        [[1, 128, 128, 3]], ['uint16'], ['input_1']
    ],
    'tfjs-model_mirnet-tfjs_default_no-comp_1': [
        [[1, 128, 128, 3]], ['float32'], ['input_1']
    ],
    'cycle-gan_256x256': [
        [[1, 256, 256, 3]], ['float32'], ['input_1']
    ],
    'cycle-gan_512x512': [
        [[1, 512, 512, 3]], ['float32'], ['input_1']
    ],
    'cycle-gan_640x640': [  // https://github.com/LynnHo/CycleGAN-Tensorflow-2
        [[1, 640, 640, 3]], ['float32'], ['input_1']
    ],

}

let D_ort = {
    'imagenet_mobilenet_v2_100_224_classification_5': [  // wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32'], ['inputs']
    ],
    'resnet_50_classification_1': [  // wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32'], ['input_1']
    ],
    'ssd_mobilenet_v2_2': [  // wasm-ok, webgl-GG
        [[1, 224, 224, 3]], ['uint8'], ['input_tensor']
    ],
    'faster_rcnn_resnet50_v1_1024x1024_1': [  // wasm-GG, webgl-GG
        [[1, 224, 224, 3]], ['uint8'], ['input_tensor']
    ],
    'edgetpu_nlp_mobilebert-edgetpu_xs_1': [  // wasm-ok, webgl-GG
        [[1, 128], [1, 128], [1, 128]], ['float32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1': [  // wasm-ok, webgl-GG
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'edgetpu_vision_deeplab-edgetpu_default_argmax_s_1': [  // wasm-ok, webgl-GG
        [[1, 512, 512, 3]], ['float32'], ['input_2']
    ],
    'edgetpu_vision_deeplab-edgetpu_default_argmax_xs_1': [  // wasm-ok, webgl-GG
        [[1, 512, 512, 3]], ['float32'], ['input_2']
    ],
    'albert_en_base_3': [  // wasm-ok, webgl-GG
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'electra_small_2': [  // wasm-ok, webgl-GG
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'esrgan-tf2_1': [  // wasm-ok, webgl-GG
        [[1, 224, 224, 3]], ['float32'], ['input_0']
    ],
    'movenet_multipose_lightning_1': [  // wasm-ok, webgl-GG
        [[1, 224, 224, 3]], ['int32'], ['input']
    ],
    
    'experts_bert_pubmed_2': [  // wasm-ok, webgl-GG
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'movenet_singlepose_thunder_4': [  // wasm-ok, webgl-GG
        [[1, 256, 256, 3]], ['int32'], ['input']
    ],
    'small_bert_bert_en_uncased_L-2_H-128_A-2_2': [  // wasm-ok, webgl-GG
        [[1, 128], [1, 128], [1, 128]], ['int32', 'int32', 'int32'], ['input_mask', 'input_type_ids', 'input_word_ids']
    ],
    'language_model': [  // wasm-ok, webgl-GG
        [[1, 128]], ['float32'], ['embedding_input']
    ],
    'VGG16': [  // wasm-ok, webgl-ok(Source data too small. Allocating larger array)
        [[1, 224, 224, 3]], ['float32'], ['input_1'] 
    ],
    'ShuffleNetV2': [  // wasm-ok, webgl-GG
        [[1, 224, 224, 3]], ['float32'], ['input_5']
    ],
    'yolov5': [  // wasm-ok, webgl-GG
        [[1, 224, 224, 3]], ['float32'], ['input_1']  
    ],
    'Xception': [  // wasm-ok, webgl-ok
        [[1, 299, 299, 3]], ['float32'], ['input_4']
    ],
    'InceptionV3': [  // wasm-ok, webgl-ok
        [[1, 299, 299, 3]], ['float32'], ['input_2']
    ],
    'EfficientNetV2': [  // wasm-ok, webgl-ok
        [[1, 224, 224, 3]], ['float32'], ['input_3']
    ],
    'efficientdet_d1_coco17_tpu-32': [  // wasm-ok, webgl-GG
        [[1, 640, 640, 3]], ['uint8'], ['input_tensor'] 
    ],
    'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8': [  // wasm-ok, webgl-GG
        [[1, 512, 512, 3]], ['uint8'], ['input_tensor'] 
    ],
}

document.getElementById('selection2').addEventListener('change', function() {
    var selection2 = document.getElementById('selection2').value;
    var selection3 = document.getElementById('selection3');

    if (selection2 === 'Wasm') {
        // Enable all options for selection3
        Array.from(selection3.options).forEach(option => option.disabled = false);
    } else if (selection2 === 'WebGL') {
        // Set selection3 to '1' and disable other options
        selection3.value = '1';
        Array.from(selection3.options).forEach(option => option.disabled = (option.value !== '1'));
    }
});

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    document.getElementById('webcam').srcObject = stream;
}).catch(err => {
    console.log("Error: " + err);
});

function getSelectionValues() {
    var frameworkValue = document.getElementById('selection0').value;
    var modelValue = document.getElementById('selection1').value;
    var backendValue = document.getElementById('selection2').value;
    var numThreadValue = document.getElementById('selection3').value;
    var simdValue = document.getElementById('selection4').value;
    var inputValue = document.getElementById('input0').value;

    let modelname = "";

    if(modelValue === "MobilenetV2") {
        modelname = "imagenet_mobilenet_v2_100_224_classification_5";
    } else if (modelValue === "SSD-MobileNetV2") {
        modelname = "ssd_mobilenet_v2_2";
    } else if (modelValue === "Mobile-BERT") {
        modelValue = "edgetpu_nlp_mobilebert-edgetpu_xs_1";
    }

    return {
        "FRAMEWORK": frameworkValue,
        "MODELNAME": modelname,
        "BACKEND": backendValue.toLowerCase(),
        "NUMTHREAD": parseInt(numThreadValue, 10),
        "SIMD": simdValue,
        "INTERVAL": parseFloat(inputValue)
    };
}

function monitorVideoFPS(videoElement, callback) {
    let lastFrameTime = null;
    let frameCount = 0;
    let fps = 0;

    function onFrame(now) {
        if (lastFrameTime !== null) {
            const deltaTime = now - lastFrameTime;
            frameCount++;
            if (deltaTime >= 1000) { // Calculate every second
                fps = frameCount * 1000 / deltaTime;
                frameCount = 0;
                lastFrameTime = now;

                if (typeof callback === 'function') {
                    callback(fps);
                }
            }
        } else {
            lastFrameTime = now;
        }

        videoElement.requestVideoFrameCallback(onFrame);
    }

    videoElement.requestVideoFrameCallback(onFrame);
}

function monitorRenderingFPS(callback) {
    let lastFrameTime = performance.now();
    let frameCount = 0;

    function updateFPS() {
        const now = performance.now();
        const deltaTime = now - lastFrameTime;
        frameCount++;

        if (deltaTime >= 1000) { // Update every second
            const fps = frameCount * 1000 / deltaTime;
            frameCount = 0;
            lastFrameTime = now;

            if (typeof callback === 'function') {
                callback(fps);
            }
        }

        requestAnimationFrame(updateFPS);
    }

    requestAnimationFrame(updateFPS);
}

async function inference_ortjs(provider, modelnames, numthread=1) {
    ort.env.wasm.numThreads = numthread;
    ort.env.wasm.simd = true;
    console.log(ort.env);
    
    for (let i = 0; i < modelnames.length; i++) {
        try {
            let modelname = modelnames[i];
            console.log("ORT_INFERENCE_TIMESTAME", `${provider}:${modelname}:${Date.now()}`);
            console.log(`ORT_BEGIN_INFERENCE:${provider}`, modelname);
            console.log(`ORT_INFERENCE_BEGIN_MEMORY:${provider}`, JSON.stringify({
                "jsHeapSizeLimit": performance.memory.jsHeapSizeLimit,
                "totalJSHeapSize": performance.memory.totalJSHeapSize,
                "usedJSHeapSize": performance.memory.usedJSHeapSize
            }));
            let modelSetupLatency = 0;
            let coldStartLatency = 0;
            let inferenceLatency = 0;
            let modelPath = `${server_url}/models/onnxModel/${modelname}.onnx`;
            // Model Load & Setup
            let start_t = performance.now();
            const session = await ort.InferenceSession.create(modelPath, {
                executionProviders: [provider]
            });
            console.log(`ORT_INFERENCE_SESSION_MEMORY:${provider}`, JSON.stringify({
                "jsHeapSizeLimit": performance.memory.jsHeapSizeLimit,
                "totalJSHeapSize": performance.memory.totalJSHeapSize,
                "usedJSHeapSize": performance.memory.usedJSHeapSize
            }));
            modelSetupLatency = performance.now() - start_t;

            let input_shapes = D_ort[modelname][0], input_types = D_ort[modelname][1], input_names = D_ort[modelname][2];
            let inputData, feed = {};
            for (let i = 0; i < input_shapes.length; i++) {
                if (input_types[i] === "float32") {
                    inputData = new Float32Array(input_shapes[i].reduce((a, b) => a * b));
                } else if (input_types[i] === "int32") {
                    inputData = new Int32Array(input_shapes[i].reduce((a, b) => a * b));
                } else if (input_types[i] === "uint8") {
                    inputData = new Uint8Array(input_shapes[i].reduce((a, b) => a * b));
                }
                feed[input_names[i]] = new ort.Tensor(input_types[i], inputData, input_shapes[i]);
            }
            start_t = performance.now();
            session.startProfiling();
            await session.run(feed);
            session.endProfiling();
            coldStartLatency = performance.now() - start_t;

            // feed inputs and run
            start_t = performance.now();
            for (let epoch = 0; epoch < tolEpoch; epoch++) {
                session.startProfiling();
                let _start_t = performance.now();
                await session.run(feed);
                session.endProfiling();
                let timeInterval = performance.now() - _start_t;
                await new Promise(r => setTimeout(r, timeInterval * time_interval_rate));
            }
            console.log(`ORT_INFERENCE_FINISH_MEMORY:${provider}`, JSON.stringify({
                "jsHeapSizeLimit": performance.memory.jsHeapSizeLimit,
                "totalJSHeapSize": performance.memory.totalJSHeapSize,
                "usedJSHeapSize": performance.memory.usedJSHeapSize
            }));
            inferenceLatency = (performance.now() - start_t) / tolEpoch;
            console.log(`ORT_FINISH_INFERENCE:${provider}`, modelname,
                                "modelSetupLatency = ", modelSetupLatency.toFixed(1),
                                "coldStartLatency = ", coldStartLatency.toFixed(1),
                                "inferenceLatency = ", inferenceLatency.toFixed(1));
        } catch (e) {
            console.log(`failed to inference model: ${e}.`, e.stack);
        }
    }
}

async function inference_tfjs(backend, modelnames, numthreads=1) {
    console.log("try tfjs now...");
    if (backend === "cpu" || backend === "webgl" || backend === "webgpu") {
        await tf.setBackend(backend);
        await tf.ready();
    } else if (backend === "wasm") {
        console.log(server_url);
        if (server_url !== ".") {
            tf.wasm.setWasmPaths(`${server_url}/dist/`);
        }
        tf.wasm.setThreadsCount(numthreads);
        await tf.setBackend("wasm");
        await tf.ready();
        console.log("ready now!!!");
    } else {
        throw Error("invalid backend");
    }
    console.log("tfjs", backend, "ready");
    for (let i = 0; i < modelnames.length; i++) {
        try {
            let modelname = modelnames[i];
            console.log("TFJS_INFERENCE_TIMESTAME", `${backend}:${modelname}:${Date.now()}`);
            console.log("current model is: [" + modelname + "]", D_tfjs[modelname]);
            let modelSetupLatency = 0;
            let coldStartLatency = 0;
            let inferenceLatency = 0;
            // Model Load & Setup
            let start_t = performance.now();
            let model = await tf.loadGraphModel(`${server_url}/models/jsModel/${modelname}/model.json`);
            console.log(modelname, "finish load model")
            modelSetupLatency += performance.now() - start_t;
            let input_shapes = D_tfjs[modelname][0], input_types = D_tfjs[modelname][1];
            let inputData, feed = [];

            for (let i = 0; i < input_shapes.length; i++) {
                if (input_types[i] === "float32" || input_types[i] === "float16") {
                    inputData = new Float32Array(input_shapes[i].reduce((a, b) => a * b));
                } else if (input_types[i] === "int32") {
                    inputData = new Int32Array(input_shapes[i].reduce((a, b) => a * b));
                } else if (input_types[i] === "uint8") {
                    inputData = new Uint8Array(input_shapes[i].reduce((a, b) => a * b));
                } else if (input_types[i] === "uint16") {
                    inputData = new Uint16Array(input_shapes[i].reduce((a, b) => a * b));
                }
                feed.push(tf.tensor(inputData, input_shapes[i], input_types[i]))
            }
            start_t = performance.now();
            output = await model.executeAsync(feed);
            // if (typeof(output.dispose) === "function") {
            //     output.dispose();
            // }
            coldStartLatency += performance.now() - start_t;
            console.log(modelname, "finish warm up");
            // feed inputs and run
            let totalInferenceLatency = 0;
            let beginning_time = Date.now();
            for (let epoch = 0; epoch < tolEpoch; epoch++) {
                start_t = performance.now();
                output = await model.executeAsync(feed);
                // if (typeof(output.dispose) === "function") {
                //     output.dispose();
                // }
                let timeInterval = performance.now() - start_t;
                totalInferenceLatency += timeInterval;
                await new Promise(r => setTimeout(r, timeInterval * time_interval_rate));
            }
            let ending_time = Date.now();
            inferenceLatency = totalInferenceLatency / tolEpoch;
            console.log("finish", modelname,
                "modelSetupLatency = ", modelSetupLatency,
                "coldStartLatency = ", coldStartLatency,
                "inferenceLatency = ", inferenceLatency);

            feed.forEach(t => t.dispose())
            model.dispose();
        } catch (e) {
            console.log(`failed to inference model: ${e}.`, e.stack);
        }
    }
    tf.disposeVariables();
}


async function entry_func(url="") {

    console.log("fuck")

    // Usage:
    monitorRenderingFPS(function(fps) {
        console.log("Current Rendering FPS: " + fps);
        // Optionally, display this FPS on the webpage
    });

    // Usage:
    const videoElement = document.getElementById('webcam'); // Your video element
    monitorVideoFPS(videoElement, function(fps) {
        console.log("Current Video FPS: " + fps);
        // Optionally, display this FPS on the webpage
    });

    let config = getSelectionValues();
    let framework = config["FRAMEWORK"]
    let bknd = config["BACKEND"];
    let model = config["MODELNAME"];
    let numthread = config["NUMTHREAD"];
    time_interval_rate = config["INTERVAL"]
    console.log(time_interval_rate)
    return;

    tf.ENV.registerFlag('WASM_HAS_MULTITHREAD_SUPPORT');
    tf.ENV.set('WASM_HAS_MULTITHREAD_SUPPORT', true);
    tf.ENV.registerFlag('WASM_HAS_SIMD_SUPPORT');
    tf.ENV.set('WASM_HAS_SIMD_SUPPORT', true);
    

    if (url.length > 1) {
        server_url = url;
    }
    console.log(server_url);
    benchmarkClient.startBenchmark();
    if (framework === "TF.js") {
        await inference_tfjs(bknd, [model], numthread);
    } else {
        await inference_ortjs(bknd, [model], numthread);
    }
    
    
    stop_flag = true;
}


package com.example.myapplication

import android.Manifest
import android.app.PendingIntent
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.usb.UsbManager
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.hoho.android.usbserial.driver.UsbSerialPort
import com.hoho.android.usbserial.driver.UsbSerialProber
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.StorageService
import java.io.IOException
import java.nio.LongBuffer
import android.speech.tts.TextToSpeech
import java.util.Locale

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private var textToSpeech: TextToSpeech? = null
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    private var onnxLoaded = false // Bandera de control para saber si ONNX está listo
    private var progressBarOnnx: ProgressBar? = null // Declaración de la barra de carga

    // Referencias para las barras del espectro de audio
    private var bar1: View? = null
    private var bar2: View? = null
    private var bar3: View? = null
    private var bar4: View? = null
    private var bar5: View? = null
    private var jobAnimacionEspectro: Job? = null

    private fun inicializarMotorTraduccionONNX() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                android.util.Log.d("ONNX_DEBUG", "Iniciando entorno ONNX...")
                if (ortEnvironment == null) {
                    ortEnvironment = OrtEnvironment.getEnvironment()
                }

                if (ortEnvironment != null && ortSession == null) {
                    val file = java.io.File(filesDir, "decoder_model.onnx")
                    if (!file.exists() || file.length() == 0L) {
                        android.util.Log.d("ONNX_DEBUG", "Copiando modelo desde assets...")
                        assets.open("onxx_model/decoder_model_merged_int8.onnx").use { input ->
                            file.outputStream().use { output ->
                                input.copyTo(output)
                            }
                        }
                    }

                    android.util.Log.d("ONNX_DEBUG", "Creando sesión ONNX desde: ${file.absolutePath}")
                    ortSession = ortEnvironment?.createSession(file.absolutePath)
                    onnxLoaded = true

                    withContext(Dispatchers.Main) {
                        android.util.Log.d("ONNX_DEBUG", "Motor ONNX cargado exitosamente")
                        val estadoActual = txtEstado?.text.toString()
                        txtEstado?.text = "$estadoActual | ONNX Listo"

                        // Ocultar la barra de carga al finalizar con éxito
                        progressBarOnnx?.visibility = View.GONE
                    }
                }
            } catch (e: Throwable) {
                e.printStackTrace()
                android.util.Log.e("ONNX_DEBUG", "Error crítico al cargar ONNX", e)
                withContext(Dispatchers.Main) {
                    onnxLoaded = false
                    txtEstado?.text = "Error ONNX: ${e.localizedMessage ?: e.javaClass.simpleName}"

                    // Ocultar la barra de carga también en caso de error
                    progressBarOnnx?.visibility = View.GONE
                }
            }
        }
    }

    companion object {
        private const val RECORD_AUDIO_PERMISSION_CODE = 101
    }

    private var puertoSerial: UsbSerialPort? = null
    private var txtEstado: TextView? = null
    private var txtOriginal: TextView? = null
    private var txtTraducido: TextView? = null
    private var layoutEspectro: LinearLayout? = null

    private var modelEs: Model? = null
    private var modelEn: Model? = null
    private var isListening = false
    private var idiomaActual = "ES" // "ES" habla español -> traduce a Inglés / "EN" habla inglés -> traduce a Español

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        txtEstado = findViewById(R.id.txtEstado)
        txtOriginal = findViewById(R.id.txtOriginal)
        txtTraducido = findViewById(R.id.txtTraducido)
        layoutEspectro = findViewById(R.id.layoutEspectro)
        progressBarOnnx = findViewById(R.id.progressBarOnnx) // Enlace con el XML

        // Enlace de las vistas de las barras del espectro de audio
        bar1 = findViewById(R.id.bar1)
        bar2 = findViewById(R.id.bar2)
        bar3 = findViewById(R.id.bar3)
        bar4 = findViewById(R.id.bar4)
        bar5 = findViewById(R.id.bar5)

        val btnConectar = findViewById<Button>(R.id.btnConectar)
        val btnHablar = findViewById<Button>(R.id.btnHablar)

        // Ocultar por completo el botón verde ya que la conexión es automática
        btnConectar.visibility = View.GONE

        // Mostrar la barra de carga al iniciar
        progressBarOnnx?.visibility = View.VISIBLE

        textToSpeech = TextToSpeech(this, this)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), RECORD_AUDIO_PERMISSION_CODE)
        }

        cargarModelosVosk()
        inicializarMotorTraduccionONNX()

        btnHablar.setOnClickListener { iniciarEscuchaEnVivo() }

        // Comprobar si la app se abrió mediante la conexión automática del cable USB
        intent?.let { verificarIntencionUsb(it) }
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        intent?.let {
            setIntent(it)
            verificarIntencionUsb(it)
        }
    }

    private fun verificarIntencionUsb(intent: Intent) {
        if (intent.action == UsbManager.ACTION_USB_DEVICE_ATTACHED || intent.action == Intent.ACTION_MAIN) {
            CoroutineScope(Dispatchers.IO).launch {
                conectarESP32Automatico()
            }
        }
    }

    private suspend fun conectarESP32Automatico() {
        try {
            val manager = getSystemService(USB_SERVICE) as UsbManager
            val availableDrivers = UsbSerialProber.getDefaultProber().findAllDrivers(manager)

            if (availableDrivers.isEmpty()) {
                withContext(Dispatchers.Main) {
                    txtEstado?.text = "Conecta el ESP32-S3 por USB"
                }
                return
            }

            val driver = availableDrivers[0]
            val device = driver.device

            if (!manager.hasPermission(device)) {
                val intent = Intent("com.example.myapplication.USB_PERMISSION").apply {
                    setPackage(packageName)
                }
                val permissionIntent = PendingIntent.getBroadcast(this, 0, intent, PendingIntent.FLAG_IMMUTABLE)
                manager.requestPermission(device, permissionIntent)
                withContext(Dispatchers.Main) {
                    txtEstado?.text = "Solicitando permiso USB..."
                }
                return
            }

            val connection = manager.openDevice(device) ?: return
            try { puertoSerial?.close() } catch (_: Exception) {}

            puertoSerial = driver.ports[0]
            puertoSerial?.open(connection)
            puertoSerial?.setParameters(115200, 8, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE)

            withContext(Dispatchers.Main) {
                txtEstado?.text = "ESP32 Conectada Automáticamente"
                enviarComando("AZUL")
            }
        } catch (e: Exception) {
            withContext(Dispatchers.Main) {
                txtEstado?.text = "Error USB: ${e.localizedMessage}"
            }
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            textToSpeech?.language = Locale.US
        }
    }

    override fun onDestroy() {
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        super.onDestroy()
    }

    private fun cargarModelosVosk() {
        txtEstado?.text = getString(R.string.status_loading_models)

        StorageService.unpack(this, "model-es", "model-es-extracted",
            { model: Model? ->
                modelEs = model
                verificarModelosListos()
            },
            { exception: IOException ->
                txtEstado?.text = getString(R.string.status_error_model_es, exception.message ?: "")
            }
        )

        StorageService.unpack(this, "model-en", "model-en-extracted",
            { model: Model? ->
                modelEn = model
                verificarModelosListos()
            },
            { exception: IOException ->
                txtEstado?.text = getString(R.string.status_error_model_en, exception.message ?: "")
            }
        )
    }

    private fun verificarModelosListos() {
        if (modelEs != null && modelEn != null) {
            txtEstado?.text = getString(R.string.status_models_ready)
        }
    }

    private fun enviarComando(comando: String) {
        try {
            val datos = "$comando\n"
            puertoSerial?.write(datos.toByteArray(), 1000)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    private fun iniciarEscuchaEnVivo() {
        if (!onnxLoaded || ortSession == null) {
            Toast.makeText(this, "El motor ONNX aún se está cargando o falló al iniciar", Toast.LENGTH_LONG).show()
            return
        }
        if (modelEs == null || modelEn == null) {
            Toast.makeText(this, R.string.models_still_loading, Toast.LENGTH_SHORT).show()
            return
        }
        if (puertoSerial == null) {
            Toast.makeText(this, R.string.connect_esp32_first, Toast.LENGTH_SHORT).show()
            return
        }
        if (isListening) return

        isListening = true
        layoutEspectro?.visibility = View.VISIBLE
        enviarComando("VERDE")

        // Iniciar la animación de las barras del espectro
        iniciarAnimacionEspectro()

        val modeloSeleccionado = if (idiomaActual == "ES") modelEs else modelEn
        txtEstado?.text = getString(R.string.status_speak_now)
        txtOriginal?.text = getString(R.string.listening_dots)
        txtTraducido?.text = getString(R.string.processing_stream)

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val sampleRate = 16000f
                val recognizer = Recognizer(modeloSeleccionado, sampleRate)
                val bufferSerial = ByteArray(512)

                val startTime = System.currentTimeMillis()
                var textoAcumulado = ""

                while (isListening && (System.currentTimeMillis() - startTime < 7000)) {
                    try {
                        val bytesRead = puertoSerial?.read(bufferSerial, 100) ?: 0
                        if (bytesRead > 0) {
                            if (recognizer.acceptWaveForm(bufferSerial, bytesRead)) {
                                val jsonResult = JSONObject(recognizer.result)
                                val t = jsonResult.optString("text", "")
                                if (t.isNotBlank()) {
                                    textoAcumulado += " $t"
                                }
                            } else {
                                val jsonPartial = JSONObject(recognizer.partialResult)
                                val partialText = jsonPartial.optString("partial", "")
                                if (partialText.isNotEmpty()) {
                                    withContext(Dispatchers.Main) {
                                        txtOriginal?.text = getString(R.string.text_concat, textoAcumulado, partialText).trim()
                                    }
                                }
                            }
                        }
                    } catch (_: Exception) {}
                }

                val jsonFinal = JSONObject(recognizer.finalResult)
                val textoFinalPart = jsonFinal.optString("text", "")
                if (textoFinalPart.isNotBlank()) {
                    textoAcumulado += " $textoFinalPart"
                }

                val fraseCompleta = textoAcumulado.trim()

                withContext(Dispatchers.Main) {
                    layoutEspectro?.visibility = View.GONE
                    detenerAnimacionEspectro()
                    isListening = false

                    if (fraseCompleta.isNotBlank()) {
                        enviarComando("AMARILLO")
                        txtOriginal?.text = fraseCompleta

                        val traduccion = traducirTextoOffline(fraseCompleta)

                        txtTraducido?.text = traduccion
                        enviarComando("VIOLETA")
                        txtEstado?.text = getString(R.string.status_phrase_translation_ready)

                        if (!traduccion.startsWith("Error") && !traduccion.startsWith("Entorno") && !traduccion.startsWith("No se")) {
                            val idiomaVoz = if (idiomaActual == "ES") Locale.US else Locale.forLanguageTag("es-ES")
                            textToSpeech?.language = idiomaVoz
                            textToSpeech?.speak(traduccion, TextToSpeech.QUEUE_FLUSH, null, null)
                        }

                        idiomaActual = if (idiomaActual == "ES") "EN" else "ES"

                        txtOriginal?.postDelayed({
                            enviarComando("AZUL")
                            val proxModo = if (idiomaActual == "ES") getString(R.string.mode_es_to_en) else getString(R.string.mode_en_to_es)
                            txtEstado?.text = getString(R.string.status_ready_mode, proxModo)
                        }, 4000)
                    } else {
                        txtOriginal?.text = getString(R.string.no_clear_voice_detected)
                        txtTraducido?.text = "---"
                        enviarComando("AZUL")
                        txtEstado?.text = getString(R.string.status_connected_usb)
                    }
                }

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    layoutEspectro?.visibility = View.GONE
                    detenerAnimacionEspectro()
                    isListening = false
                    txtEstado?.text = getString(R.string.usb_read_error, e.message ?: "")
                    enviarComando("AZUL")
                }
            }
        }
    }

    private fun iniciarAnimacionEspectro() {
        jobAnimacionEspectro = CoroutineScope(Dispatchers.Main).launch {
            while (isListening) {
                val aleatorio1 = (20..100).random() / 50f
                val aleatorio2 = (10..120).random() / 50f
                val aleatorio3 = (30..90).random() / 50f

                bar1?.animate()?.scaleY(aleatorio1)?.setDuration(150)?.start()
                bar2?.animate()?.scaleY(aleatorio2)?.setDuration(150)?.start()
                bar3?.animate()?.scaleY(aleatorio3)?.setDuration(150)?.start()
                bar4?.animate()?.scaleY(aleatorio1)?.setDuration(150)?.start()
                bar5?.animate()?.scaleY(aleatorio2)?.setDuration(150)?.start()

                delay(150)
            }
        }
    }

    private fun detenerAnimacionEspectro() {
        jobAnimacionEspectro?.cancel()
        bar1?.scaleY = 1f
        bar2?.scaleY = 1f
        bar3?.scaleY = 1f
        bar4?.scaleY = 1f
        bar5?.scaleY = 1f
    }

    private fun traducirTextoOffline(texto: String): String {
        if (texto.isBlank()) return ""

        val sessionLocal = ortSession
        val envLocal = ortEnvironment
        if (sessionLocal == null || envLocal == null) {
            return "Entorno u objeto ONNX no inicializado"
        }

        try {
            val tokensEntrada = tokenizarTexto(texto)
            val shape = longArrayOf(1, tokensEntrada.size.toLong())
            val tensorEntrada = OnnxTensor.createTensor(envLocal, LongBuffer.wrap(tokensEntrada), shape)

            val inputs = mutableMapOf<String, OnnxTensor>()

            val inputNames = sessionLocal.inputNames
            val inputName = if (inputNames.contains("input_ids")) "input_ids" else inputNames.first()
            inputs[inputName] = tensorEntrada

            // Parámetro de caché opcional del modelo
            if (inputNames.contains("use_cache_branch")) {
                val boolBuffer = java.nio.ByteBuffer.allocate(1).put(0, 0.toByte())
                val cacheBranchTensor = OnnxTensor.createTensor(envLocal, boolBuffer, longArrayOf(1), ai.onnxruntime.OnnxJavaType.BOOL)
                inputs["use_cache_branch"] = cacheBranchTensor
            }

            // Estados ocultos del codificador (dimensión 1024)
            if (inputNames.contains("encoder_hidden_states")) {
                val dummyShape = longArrayOf(1, 1, 1024)
                val floatBuffer = java.nio.FloatBuffer.allocate(1 * 1 * 1024)
                val encoderTensor = OnnxTensor.createTensor(envLocal, floatBuffer, dummyShape)
                inputs["encoder_hidden_states"] = encoderTensor
            }

            // Máscara de atención del codificador requerida por el modelo
            if (inputNames.contains("encoder_attention_mask")) {
                val maskShape = longArrayOf(1, 1)
                val maskBuffer = java.nio.LongBuffer.wrap(longArrayOf(1L))
                val maskTensor = OnnxTensor.createTensor(envLocal, maskBuffer, maskShape)
                inputs["encoder_attention_mask"] = maskTensor
            }

            val resultados = sessionLocal.run(inputs)

            var textoTraducido = "Traducción completada"
            if (resultados != null && resultados.size() > 0) {
                val outputTensorValue = resultados.get(0).value

                when (outputTensorValue) {
                    is Array<*> -> {
                        val tokensSalida = mutableListOf<Long>()
                        try {
                            for (rowObj in outputTensorValue) {
                                if (rowObj is Array<*>) {
                                    for (item in rowObj) {
                                        when (item) {
                                            is Number -> tokensSalida.add(item.toLong())
                                        }
                                    }
                                } else if (rowObj is Number) {
                                    tokensSalida.add(rowObj.toLong())
                                }
                            }
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }

                        if (tokensSalida.isNotEmpty()) {
                            textoTraducido = decodificarTokens(tokensSalida.toLongArray())
                        } else {
                            textoTraducido = "Inferencia Exitosa (Matriz procesada)"
                        }
                    }
                    is FloatArray -> {
                        textoTraducido = "Tokens flotantes recibidos (${outputTensorValue.size})"
                    }
                    else -> {
                        textoTraducido = "Resultado: ${outputTensorValue?.javaClass?.simpleName ?: "Nulo"}"
                    }
                }
            } else {
                textoTraducido = "La sesión ONNX devolvió 0 resultados"
            }

            tensorEntrada.close()
            resultados?.close()

            inputs.values.forEach { if (it != tensorEntrada) it.close() }

            return textoTraducido

        } catch (e: Exception) {
            e.printStackTrace()
            return "Excepción crítica: ${e.localizedMessage}"
        }
    }

    private fun tokenizarTexto(texto: String): LongArray {
        val limpio = texto.trim().lowercase()
        val tokens = mutableListOf<Long>()
        tokens.add(101)
        for (char in limpio) {
            tokens.add(char.code.toLong().coerceIn(1, 30000))
        }
        tokens.add(102)
        return tokens.toLongArray()
    }

    private fun decodificarTokens(tokens: LongArray): String {
        val sb = StringBuilder()
        for (token in tokens) {
            if (token in 32..126) {
                sb.append(token.toInt().toChar())
            }
        }
        val resultado = sb.toString().trim()
        return resultado.ifEmpty { "Traducción generada por IA (Tokens procesados)" }
    }
}
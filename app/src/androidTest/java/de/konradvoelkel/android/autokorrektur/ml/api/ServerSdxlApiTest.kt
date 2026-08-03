package de.konradvoelkel.android.autokorrektur.ml.api

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream

@RunWith(AndroidJUnit4::class)
class ServerSdxlApiTest : AndroidInstrumentedBaseTest() {

    private lateinit var server: MockWebServer
    private lateinit var api: ServerSdxlApi

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()

        // We need to inject the URL into ServerSdxlApi. 
        // Currently it's hardcoded to BuildConfig.BACKEND_URL.
        // For testing, we might need a way to override it.
        // Let's check ServerSdxlApi.kt again.
        api = ServerSdxlApi(appContext)
    }

    @After
    fun tearDown() {
        server.shutdown()
    }

    @Test
    fun testProcessWithSdxl_success() {
        runBlocking {
            // Prepare mock response (a small 1x1 red JPEG)
            val out = ByteArrayOutputStream()
            val mockBitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
            mockBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
            val responseBytes = out.toByteArray()

            server.enqueue(MockResponse().setBody(okio.Buffer().write(responseBytes)))

            // We need to override the URL. Since we can't easily change BuildConfig, 
            // let's use reflection to change the serverUrl field in the api instance.
            val field = ServerSdxlApi::class.java.getDeclaredField("serverUrl")
            field.isAccessible = true
            field.set(api, server.url("/v1/inpaint").toString())

            val orig = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)
            val mask = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)
            val prev = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)

            val result = api.processWithSdxl(orig, mask, prev)

            assertNotNull(result)
            assertEquals(1, result.width)
            assertEquals(1, result.height)

            val request = server.takeRequest()
            assertEquals("POST", request.method)
            assertEquals("/v1/inpaint", request.path)
        }
    }

    @Test(expected = Exception::class)
    fun testProcessWithSdxl_error() {
        runBlocking {
            server.enqueue(MockResponse().setResponseCode(500).setBody("Internal Server Error"))

            val field = ServerSdxlApi::class.java.getDeclaredField("serverUrl")
            field.isAccessible = true
            field.set(api, server.url("/v1/inpaint").toString())

            val orig = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)
            val mask = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)
            val prev = Bitmap.createBitmap(10, 10, Bitmap.Config.ARGB_8888)

            api.processWithSdxl(orig, mask, prev)
        }
    }
}

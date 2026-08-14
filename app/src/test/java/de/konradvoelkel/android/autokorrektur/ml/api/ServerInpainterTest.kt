package de.konradvoelkel.android.autokorrektur.ml.api

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import de.konradvoelkel.android.autokorrektur.manager.QuotaManager
import de.konradvoelkel.android.autokorrektur.ml.errors.CloudInferenceException
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.mockkStatic
import io.mockk.unmockkAll
import kotlinx.coroutines.runBlocking
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.After
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class ServerInpainterTest {

    private lateinit var mockContext: Context
    private lateinit var mockQuotaManager: QuotaManager
    private lateinit var mockClient: OkHttpClient
    private lateinit var mockCall: Call
    private lateinit var dummyBitmap: Bitmap

    @Before
    fun setUp() {
        mockkObject(AppLogger)
        every { AppLogger.info(any(), any()) } returns Unit
        every { AppLogger.error(any(), any()) } returns Unit
        every { AppLogger.debug(any(), any()) } returns Unit
        every { AppLogger.warn(any(), any()) } returns Unit

        mockkStatic(BitmapFactory::class)

        mockContext = mockk(relaxed = true)
        mockQuotaManager = mockk(relaxed = true)
        mockClient = mockk(relaxed = true)
        mockCall = mockk(relaxed = true)

        dummyBitmap = mockk(relaxed = true)
        every { dummyBitmap.compress(any(), any(), any()) } answers {
            val stream = arg<java.io.OutputStream>(2)
            stream.write(ByteArray(64) { 0xFF.toByte() })
            true
        }

        every { mockQuotaManager.hasAvailableQuota() } returns true
        every { mockQuotaManager.getDeviceUuid() } returns "uuid-1234"
        every { mockClient.newCall(any()) } returns mockCall
    }

    @After
    fun tearDown() {
        unmockkAll()
    }

    @Test
    fun testServerInpainter_successReturnsBitmap() = runBlocking {
        val dummyRequest = Request.Builder().url("http://10.0.2.2:8000/v1/inpaint").build()
        val decodedBitmap = mockk<Bitmap>(relaxed = true)
        every { BitmapFactory.decodeByteArray(any(), any(), any()) } returns decodedBitmap

        val successResponse = Response.Builder()
            .request(dummyRequest)
            .protocol(Protocol.HTTP_1_1)
            .code(200)
            .message("OK")
            .body(ByteArray(32).toResponseBody("image/jpeg".toMediaTypeOrNull()))
            .build()

        every { mockCall.execute() } returns successResponse

        val inpainter: ServerInpainter = ServerSdxlApi(
            context = mockContext,
            client = mockClient,
            quotaManager = mockQuotaManager
        )

        val result = inpainter.processWithSdxl(dummyBitmap, dummyBitmap, dummyBitmap)
        assertNotNull(result)
    }

    @Test
    fun testServerInpainter_400BadRequest_throwsCloudInferenceException() = runBlocking {
        val dummyRequest = Request.Builder().url("http://10.0.2.2:8000/v1/inpaint").build()
        val errorResponse = Response.Builder()
            .request(dummyRequest)
            .protocol(Protocol.HTTP_1_1)
            .code(400)
            .message("Bad Request")
            .body("Invalid image payload".toResponseBody("text/plain".toMediaTypeOrNull()))
            .build()

        every { mockCall.execute() } returns errorResponse

        val inpainter: ServerInpainter = ServerSdxlApi(
            context = mockContext,
            client = mockClient,
            quotaManager = mockQuotaManager
        )

        val ex = assertThrows(CloudInferenceException::class.java) {
            runBlocking {
                inpainter.processWithSdxl(dummyBitmap, dummyBitmap, dummyBitmap)
            }
        }
        assertTrue(ex.message!!.contains("400"))
    }
}

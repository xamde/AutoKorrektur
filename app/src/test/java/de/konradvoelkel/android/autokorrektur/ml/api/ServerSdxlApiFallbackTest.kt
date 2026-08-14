package de.konradvoelkel.android.autokorrektur.ml.api

import android.content.Context
import android.graphics.Bitmap
import de.konradvoelkel.android.autokorrektur.manager.QuotaManager
import de.konradvoelkel.android.autokorrektur.ml.errors.CloudInferenceException
import de.konradvoelkel.android.autokorrektur.ml.errors.QuotaExceededException
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.runBlocking
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.IOException
import java.net.ConnectException
import java.net.SocketTimeoutException

import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import io.mockk.mockkObject

class ServerSdxlApiFallbackTest {

    private lateinit var mockContext: Context
    private lateinit var mockQuotaManager: QuotaManager
    private lateinit var mockOkHttpClient: OkHttpClient
    private lateinit var mockCall: Call
    private lateinit var dummyBitmap: Bitmap

    @Before
    fun setUp() {
        mockkObject(AppLogger)
        every { AppLogger.info(any(), any()) } returns Unit
        every { AppLogger.error(any(), any()) } returns Unit
        every { AppLogger.debug(any(), any()) } returns Unit
        every { AppLogger.warn(any(), any()) } returns Unit

        mockContext = mockk(relaxed = true)
        mockQuotaManager = mockk(relaxed = true)
        mockOkHttpClient = mockk(relaxed = true)
        mockCall = mockk(relaxed = true)

        dummyBitmap = mockk(relaxed = true)
        every { dummyBitmap.compress(any(), any(), any()) } answers {
            val stream = arg<java.io.OutputStream>(2)
            stream.write(ByteArray(32) { 1 })
            true
        }

        every { mockQuotaManager.hasAvailableQuota() } returns true
        every { mockQuotaManager.getDeviceUuid() } returns "test-device-uuid-1234"
        every { mockOkHttpClient.newCall(any()) } returns mockCall
    }

    @Test
    fun processWithSdxl_whenNetworkTimesOut_throwsCloudInferenceException_andPreservesQuota() = runBlocking {
        every { mockCall.execute() } throws SocketTimeoutException("Connection timed out to 10.0.2.2")

        val api = ServerSdxlApi(
            context = mockContext,
            client = mockOkHttpClient,
            quotaManager = mockQuotaManager
        )

        val exception = assertThrows(CloudInferenceException::class.java) {
            runBlocking {
                api.processWithSdxl(dummyBitmap, dummyBitmap, dummyBitmap)
            }
        }

        assertTrue(exception.message!!.contains("Cloud communication error"))
        // CRITICAL: Quota must NOT be consumed if network request failed
        verify(exactly = 0) { mockQuotaManager.consumeQuota() }
    }

    @Test
    fun processWithSdxl_whenHostUnreachableOrRefused_throwsCloudInferenceException_andPreservesQuota() = runBlocking {
        every { mockCall.execute() } throws ConnectException("Failed to connect to /10.0.2.2:8000")

        val api = ServerSdxlApi(
            context = mockContext,
            client = mockOkHttpClient,
            quotaManager = mockQuotaManager
        )

        val exception = assertThrows(CloudInferenceException::class.java) {
            runBlocking {
                api.processWithSdxl(dummyBitmap, dummyBitmap, dummyBitmap)
            }
        }

        assertTrue(exception.message!!.contains("Failed to connect"))
        verify(exactly = 0) { mockQuotaManager.consumeQuota() }
    }

    @Test
    fun processWithSdxl_whenServerReturns503_throwsCloudInferenceException_andPreservesQuota() = runBlocking {
        val dummyRequest = Request.Builder().url("http://10.0.2.2:8000/v1/inpaint").build()
        val errorResponse = Response.Builder()
            .request(dummyRequest)
            .protocol(Protocol.HTTP_1_1)
            .code(503)
            .message("Service Unavailable")
            .body("SDXL GPU worker busy".toResponseBody("text/plain".toMediaTypeOrNull()))
            .build()

        every { mockCall.execute() } returns errorResponse

        val api = ServerSdxlApi(
            context = mockContext,
            client = mockOkHttpClient,
            quotaManager = mockQuotaManager
        )

        val exception = assertThrows(CloudInferenceException::class.java) {
            runBlocking {
                api.processWithSdxl(dummyBitmap, dummyBitmap, dummyBitmap)
            }
        }

        assertTrue(exception.message!!.contains("503"))
        verify(exactly = 0) { mockQuotaManager.consumeQuota() }
    }

    @Test
    fun processWithSdxl_whenQuotaExceeded_throwsQuotaExceededException_withoutNetworkCall() = runBlocking {
        every { mockQuotaManager.hasAvailableQuota() } returns false

        val api = ServerSdxlApi(
            context = mockContext,
            client = mockOkHttpClient,
            quotaManager = mockQuotaManager
        )

        assertThrows(QuotaExceededException::class.java) {
            runBlocking {
                api.processWithSdxl(dummyBitmap, dummyBitmap, dummyBitmap)
            }
        }

        verify(exactly = 0) { mockOkHttpClient.newCall(any()) }
        verify(exactly = 0) { mockQuotaManager.consumeQuota() }
    }
}

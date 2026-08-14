package de.konradvoelkel.android.autokorrektur.pipeline

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.work.ListenableWorker
import androidx.work.testing.TestListenableWorkerBuilder
import androidx.work.workDataOf
import de.konradvoelkel.android.autokorrektur.shared.AndroidInstrumentedBaseTest
import io.mockk.coEvery
import io.mockk.mockkConstructor
import io.mockk.unmockkAll
import org.junit.After
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import android.graphics.Bitmap

@RunWith(AndroidJUnit4::class)
class BatchProcessingWorkerInstrumentedTest : AndroidInstrumentedBaseTest() {

    @After
    fun tearDown() {
        unmockkAll()
    }

    @Test
    fun testBatchProcessingWorker_emptyInput_fails() {
        runBlocking {
            val context = ApplicationProvider.getApplicationContext<Context>()
            val worker = TestListenableWorkerBuilder<BatchProcessingWorker>(context)
                .setInputData(workDataOf(BatchProcessingWorker.KEY_IMAGE_URIS to emptyArray<String>()))
                .build()

            val result = worker.doWork()

            assertTrue(result is ListenableWorker.Result.Failure)
            val error = (result as ListenableWorker.Result.Failure).outputData.getString(
                BatchProcessingWorker.KEY_ERROR
            )
            assertEquals("No image URIs provided", error)
        }
    }

    @Test
    fun testBatchProcessingWorker_successWithMockedPipeline() {
        runBlocking {
            val context = ApplicationProvider.getApplicationContext<Context>()

            // Mock StaticImagePipeline constructor and methods
            mockkConstructor(StaticImagePipeline::class)
            coEvery { anyConstructed<StaticImagePipeline>().initialize() } returns Unit
            coEvery { anyConstructed<StaticImagePipeline>().close() } returns Unit
            coEvery {
                anyConstructed<StaticImagePipeline>().processImage(
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any(),
                    any()
                )
            } returns PipelineResult(
                originalBitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888),
                maskBitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888),
                inpaintedBitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
            )

            val worker = TestListenableWorkerBuilder<BatchProcessingWorker>(context)
                .setInputData(
                    workDataOf(
                        BatchProcessingWorker.KEY_IMAGE_URIS to arrayOf(
                            "content://test/1",
                            "content://test/2"
                        )
                    )
                )
                .build()

            val result = worker.doWork()

            assertTrue(result is ListenableWorker.Result.Success)
            val successCount = (result as ListenableWorker.Result.Success).outputData.getInt(
                BatchProcessingWorker.KEY_SUCCESS_COUNT,
                0
            )
            assertEquals(2, successCount)
        }
    }
}

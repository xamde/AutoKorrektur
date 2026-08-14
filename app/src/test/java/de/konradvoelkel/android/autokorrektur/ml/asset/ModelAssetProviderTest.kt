package de.konradvoelkel.android.autokorrektur.ml.asset

import android.content.Context
import android.content.res.AssetManager
import io.mockk.every
import io.mockk.mockk
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.File

class ModelAssetProviderTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    @Test
    fun testOpenModelAssetFromAssetManager() {
        val context = mockk<Context>(relaxed = true)
        val assetManager = mockk<AssetManager>(relaxed = true)
        val filesDir = tempFolder.newFolder("files")

        every { context.filesDir } returns filesDir
        every { context.assets } returns assetManager
        every { assetManager.open("model/yolo.onnx") } returns ByteArrayInputStream("mock-weights".toByteArray())

        val stream = ModelAssetProvider.openModelAsset(context, "model/yolo.onnx")
        val content = stream.bufferedReader().use { it.readText() }
        assertEquals("mock-weights", content)
    }

    @Test
    fun testGetOrExtractModelFileCreatesLocalCopy() {
        val context = mockk<Context>(relaxed = true)
        val assetManager = mockk<AssetManager>(relaxed = true)
        val filesDir = tempFolder.newFolder("files_extract")

        every { context.filesDir } returns filesDir
        every { context.assets } returns assetManager
        every { assetManager.open("model/mi-gan.onnx") } returns ByteArrayInputStream("migan-bytes".toByteArray())

        val extractedFile = ModelAssetProvider.getOrExtractModelFile(context, "model/mi-gan.onnx")
        assertNotNull(extractedFile)
        assertTrue(extractedFile.exists())
        assertEquals("migan-bytes", extractedFile.readText())
    }
}

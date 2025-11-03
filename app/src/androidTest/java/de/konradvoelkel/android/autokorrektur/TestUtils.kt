@file:Suppress("unused")
package de.konradvoelkel.android.autokorrektur

import android.content.Context
import java.io.File
import java.io.IOException

/**
 * Deprecated shim kept for backward compatibility during migration.
 * Please use shared.AndroidTestUtils instead.
 */
@Deprecated("Use de.konradvoelkel.android.autokorrektur.shared.AndroidTestUtils")
object TestUtils {

    fun initOpenCV() {
        // Delegate to shared version
        de.konradvoelkel.android.autokorrektur.shared.AndroidTestUtils.initOpenCV()
    }

    @Throws(IOException::class)
    fun copyAssetToCache(context: Context, assetFileName: String): File {
        return de.konradvoelkel.android.autokorrektur.shared.AndroidTestUtils.copyAssetToCache(context, assetFileName)
    }
}

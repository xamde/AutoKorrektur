package de.konradvoelkel.android.autokorrektur.telemetry

import android.app.ActivityManager
import android.content.Context
import android.content.SharedPreferences
import android.os.Build
import androidx.core.content.edit
import de.konradvoelkel.android.autokorrektur.BuildConfig
import de.konradvoelkel.android.autokorrektur.manager.PreferencesConstants
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

/**
 * Opt-in, on-device-only diagnostics ("Diagnosedaten", PRIVACY_POLICY.md §6).
 *
 * What it is: a process-wide facade the pipelines call with small, primitive-only event maps
 * (`Telemetry.record("pipeline_run", mapOf("total_ms" to 812, ...))`). Events are appended to a
 * size-capped JSON Lines file in the app's private `filesDir` by [TelemetryStore]. Nothing is
 * ever uploaded: the only way a line leaves the device is the user sharing the export file from
 * the Diagnostics dialog. There is no server side and no network code here on purpose — if that
 * ever changes, PRIVACY_POLICY.md §6, the Play Data Safety form and the consent copy change first.
 *
 * What it records (and what it must not): timings, sizes in pixels, modes, counts, error class
 * names, device model / RAM / cores / Android version, app version, and a random install id
 * generated when the user switches diagnostics on and dropped when they delete the data. Never
 * image content, URIs, file names, exception messages (they can contain paths) or anything typed
 * by the user. Reviewers: every `record(...)` call site is a one-line grep.
 *
 * Off by default; [record] is a cheap no-op while disabled or before [initialize], so pipeline
 * code (and its JVM unit tests, which never initialize this) can call it unconditionally.
 */
object Telemetry {
    private const val DIRECTORY_NAME = "telemetry"
    private const val EXPORT_DIRECTORY_NAME = "diagnostics"
    private const val APP_PACKAGE_PREFIX = "de.konradvoelkel.android.autokorrektur."

    private var store: TelemetryStore? = null
    private var prefs: SharedPreferences? = null
    private var sessionFields: Map<String, Any?> = emptyMap()

    @Volatile
    private var enabled = false

    /** True when the user has switched diagnostics on. */
    val isEnabled: Boolean get() = enabled

    /** The random per-installation id, or null while diagnostics are off. */
    val installId: String? get() = prefs?.getString(PreferencesConstants.KEY_TELEMETRY_INSTALL_ID, null)

    /**
     * Wires the store and preferences; called once from the Application. Records a
     * `session_start` when diagnostics are already on, and installs an uncaught-exception hook
     * that records a `crash` line (exception class + first in-app stack frame, no message) before
     * handing the crash to Android's default handler.
     */
    fun initialize(context: Context) {
        val appContext = context.applicationContext
        prefs = appContext.getSharedPreferences(PreferencesConstants.PREFS_NAME, Context.MODE_PRIVATE)
        store = TelemetryStore(File(appContext.filesDir, DIRECTORY_NAME))
        sessionFields = collectSessionFields(appContext)
        enabled = prefs?.getBoolean(PreferencesConstants.KEY_TELEMETRY_ENABLED, false) ?: false
        installCrashHook()
        if (enabled) recordSessionStart()
    }

    /** Switches diagnostics on (creating the install id and recording `session_start`) or off. */
    fun setEnabled(value: Boolean) {
        val p = prefs ?: return
        if (value && installId == null) {
            p.edit { putString(PreferencesConstants.KEY_TELEMETRY_INSTALL_ID, UUID.randomUUID().toString()) }
        }
        p.edit { putBoolean(PreferencesConstants.KEY_TELEMETRY_ENABLED, value) }
        enabled = value
        AppLogger.info("Telemetry ${if (value) "enabled" else "disabled"}")
        if (value) recordSessionStart()
    }

    /**
     * Appends one event when enabled. Values must be primitives (String/Boolean/Int/Long/Float/
     * Double/null); anything else is stringified by [TelemetryEvent], so don't pass objects.
     */
    fun record(name: String, fields: Map<String, Any?> = emptyMap()) {
        if (!enabled) return
        val s = store ?: return
        try {
            s.append(TelemetryEvent(name, System.currentTimeMillis(), fields))
        } catch (e: Exception) {
            AppLogger.warn("Telemetry: failed to record '$name': ${e.javaClass.simpleName}")
        }
    }

    fun eventCount(): Int = store?.eventCount() ?: 0

    fun sizeBytes(): Long = store?.sizeBytes() ?: 0L

    /** Deletes every recorded event and the install id (a new one is made if re-enabled). */
    fun clear() {
        store?.clear()
        prefs?.edit { remove(PreferencesConstants.KEY_TELEMETRY_INSTALL_ID) }
        if (enabled) {
            // Keep the invariant "enabled ⇒ install id exists" so later lines stay attributable.
            prefs?.edit { putString(PreferencesConstants.KEY_TELEMETRY_INSTALL_ID, UUID.randomUUID().toString()) }
            recordSessionStart()
        }
        AppLogger.info("Telemetry: cleared")
    }

    /**
     * Copies the events file into the cache directory (the FileProvider `cache` path) under a
     * dated name, for the share sheet. Returns null when there is nothing to export.
     */
    fun buildExportFile(context: Context): File? {
        val source = store?.eventsFile ?: return null
        if (!source.exists() || source.length() == 0L) return null
        val dir = File(context.cacheDir, EXPORT_DIRECTORY_NAME).apply { mkdirs() }
        dir.listFiles()?.forEach { it.delete() }
        val stamp = SimpleDateFormat("yyyyMMdd-HHmm", Locale.US).format(Date())
        val target = File(dir, "autokorrektur-diagnostics-$stamp.jsonl")
        source.copyTo(target, overwrite = true)
        return target
    }

    private fun recordSessionStart() {
        record("session_start", sessionFields + ("install_id" to installId))
    }

    private fun collectSessionFields(context: Context): Map<String, Any?> {
        val memoryInfo = ActivityManager.MemoryInfo()
        (context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager)?.getMemoryInfo(memoryInfo)
        val ramGb = memoryInfo.totalMem / (1024.0 * 1024.0 * 1024.0)
        return mapOf(
            "app_version" to BuildConfig.VERSION_NAME,
            "version_code" to BuildConfig.VERSION_CODE,
            "flavor" to BuildConfig.FLAVOR,
            "build_type" to BuildConfig.BUILD_TYPE,
            "sdk_int" to Build.VERSION.SDK_INT,
            "manufacturer" to Build.MANUFACTURER,
            "model" to Build.MODEL,
            "ram_gb" to Math.round(ramGb * 10) / 10.0,
            "cores" to Runtime.getRuntime().availableProcessors(),
            "locale" to Locale.getDefault().toLanguageTag(),
        )
    }

    private fun installCrashHook() {
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        if (previous is CrashHook) return
        Thread.setDefaultUncaughtExceptionHandler(CrashHook(previous))
    }

    private class CrashHook(private val previous: Thread.UncaughtExceptionHandler?) :
        Thread.UncaughtExceptionHandler {
        override fun uncaughtException(thread: Thread, throwable: Throwable) {
            try {
                val frame = throwable.stackTrace.firstOrNull { it.className.startsWith(APP_PACKAGE_PREFIX) }
                    ?: throwable.stackTrace.firstOrNull()
                record(
                    "crash",
                    mapOf(
                        "exception" to throwable.javaClass.name,
                        "thread" to thread.name,
                        "frame" to frame?.let { "${it.className}.${it.methodName}:${it.lineNumber}" },
                    )
                )
            } catch (_: Throwable) {
                // Never let diagnostics interfere with crash handling.
            }
            previous?.uncaughtException(thread, throwable)
        }
    }
}

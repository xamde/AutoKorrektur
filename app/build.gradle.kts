import java.util.Properties
import java.io.ByteArrayOutputStream

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.detekt)
    id("jacoco")
}

android {
    namespace = "de.konradvoelkel.android.autokorrektur"
    compileSdk = 37

    val gitCommitCountProvider = providers.exec {
        commandLine("git", "rev-list", "--count", "HEAD")
        isIgnoreExitValue = true
    }.standardOutput.asText.map { text ->
        text.trim().toIntOrNull() ?: 170
    }

    val gitVersionNameProvider = providers.exec {
        commandLine("git", "describe", "--tags", "--always")
        isIgnoreExitValue = true
    }.standardOutput.asText.map { text ->
        val trimmed = text.trim()
        if (trimmed.isNotEmpty()) trimmed else "1.0.0"
    }

    defaultConfig {
        applicationId = "de.konradvoelkel.android.autokorrektur"
        minSdk = 29
        targetSdk = 36
        versionCode = gitCommitCountProvider.getOrElse(170)
        versionName = gitVersionNameProvider.getOrElse("1.0.0")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    // MVP tier feature flags (see docs/MVP_FEATURE_FLAG_PLAN.md). `core` is the only flavor
    // that ever goes to the public Play Store listing; plus/beta/full are internal/opt-in-tester
    // builds distributed as direct APKs. `full` reproduces today's pre-flavor app exactly (all
    // flags true, all 4 ABIs) so it stays the CI/dev baseline with no coverage regression.
    flavorDimensions += "scope"
    productFlavors {
        create("core") {
            dimension = "scope"
            buildConfigField("boolean", "FEATURE_LIVE_AR", "false")
            buildConfigField("boolean", "FEATURE_VIDEO_SNIPPETS", "false")
            buildConfigField("boolean", "FEATURE_CLOUD_SDXL", "false")
            buildConfigField("boolean", "FEATURE_HIGH_RES_PROGRESSIVE", "false")
            buildConfigField("boolean", "FEATURE_MANUAL_MASK_BRUSH", "false")
            buildConfigField("boolean", "FEATURE_BATCH_PROCESSING", "false")
            buildConfigField("boolean", "FEATURE_EXTRA_EXPORT_LAYOUTS", "false")
            ndk { abiFilters += "arm64-v8a" }
            // no applicationIdSuffix: this is "the app" as far as Play/users are concerned
        }
        create("plus") {
            dimension = "scope"
            applicationIdSuffix = ".plus"
            buildConfigField("boolean", "FEATURE_LIVE_AR", "false")
            buildConfigField("boolean", "FEATURE_VIDEO_SNIPPETS", "false")
            buildConfigField("boolean", "FEATURE_CLOUD_SDXL", "false")
            buildConfigField("boolean", "FEATURE_HIGH_RES_PROGRESSIVE", "false")
            buildConfigField("boolean", "FEATURE_MANUAL_MASK_BRUSH", "false")
            buildConfigField("boolean", "FEATURE_BATCH_PROCESSING", "false")
            buildConfigField("boolean", "FEATURE_EXTRA_EXPORT_LAYOUTS", "true")
            ndk { abiFilters += "arm64-v8a" }
        }
        create("beta") {
            dimension = "scope"
            applicationIdSuffix = ".beta"
            buildConfigField("boolean", "FEATURE_LIVE_AR", "false")
            buildConfigField("boolean", "FEATURE_VIDEO_SNIPPETS", "false")
            buildConfigField("boolean", "FEATURE_CLOUD_SDXL", "true")
            buildConfigField("boolean", "FEATURE_HIGH_RES_PROGRESSIVE", "true")
            buildConfigField("boolean", "FEATURE_MANUAL_MASK_BRUSH", "true")
            buildConfigField("boolean", "FEATURE_BATCH_PROCESSING", "true")
            buildConfigField("boolean", "FEATURE_EXTRA_EXPORT_LAYOUTS", "true")
            ndk { abiFilters += "arm64-v8a" }
        }
        create("full") {
            dimension = "scope"
            applicationIdSuffix = ".full"
            buildConfigField("boolean", "FEATURE_LIVE_AR", "true")
            buildConfigField("boolean", "FEATURE_VIDEO_SNIPPETS", "true")
            buildConfigField("boolean", "FEATURE_CLOUD_SDXL", "true")
            buildConfigField("boolean", "FEATURE_HIGH_RES_PROGRESSIVE", "true")
            buildConfigField("boolean", "FEATURE_MANUAL_MASK_BRUSH", "true")
            buildConfigField("boolean", "FEATURE_BATCH_PROCESSING", "true")
            buildConfigField("boolean", "FEATURE_EXTRA_EXPORT_LAYOUTS", "true")
            // no abiFilters override -> all 4 ABIs, for emulators/dev
        }
    }

    val keystorePropertiesFile = rootProject.file("keystore.properties")
    val hasReleaseKeystore = keystorePropertiesFile.exists()

    signingConfigs {
        if (hasReleaseKeystore) {
            create("release") {
                val properties = Properties()
                keystorePropertiesFile.inputStream().use { properties.load(it) }
                storeFile = rootProject.file(properties.getProperty("storeFile"))
                storePassword = properties.getProperty("storePassword")
                keyAlias = properties.getProperty("keyAlias")
                keyPassword = properties.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        debug {
            enableUnitTestCoverage = true
            enableAndroidTestCoverage = true
            buildConfigField("String", "BACKEND_URL", "\"http://127.0.0.1:8000/v1/inpaint\"")
            // Evaluation-mode dev sliders (mask upscale/downshift, score threshold, model
            // chooser) — never a real end-user feature, so this stays debug-only regardless
            // of tier flavor.
            buildConfigField("boolean", "FEATURE_EVALUATION_MODE", "true")
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            signingConfig = if (hasReleaseKeystore) signingConfigs.getByName("release") else signingConfigs.getByName("debug")
            buildConfigField(
                "String",
                "BACKEND_URL",
                "\"https://api.autokorrektur.example.com/v1/inpaint\""
            )
            buildConfigField("boolean", "FEATURE_EVALUATION_MODE", "false")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    testOptions {
        unitTests.isIncludeAndroidResources = true
    }
    lint {
        // Pre-existing debt, unrelated to the MVP feature-flag work: lintFullDebug currently
        // finds 119 MissingTranslation errors (values/strings.xml vs values-en), confirmed
        // present before this session too (verified against commit a2a81e9). Same underlying
        // issue as TESTING.md §8's StringResourceLocalizationTest, which already ratchets it as
        // known but doesn't fix it. Baselined here so CI can build/test the new flavors without
        // being blocked by unrelated debt — new lint issues introduced later still fail CI.
        baseline = file("lint-baseline.xml")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }
    buildFeatures {
        viewBinding = true
        buildConfig = true
    }
    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
            excludes += "META-INF/LICENSE.md"
            excludes += "META-INF/LICENSE-notice.md"
        }
    }
}

kotlin {
    jvmToolchain(21)
}

detekt {
    buildUponDefaultConfig = true
    allRules = false
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.exifinterface)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.navigation.fragment.ktx)
    implementation(libs.androidx.navigation.ui.ktx)
    implementation(libs.onnxruntime.android)
    implementation(libs.tensorflow.lite)
    implementation(libs.opencv)
    implementation(libs.okhttp)
    implementation(libs.androidx.work.runtime.ktx)

    // JVM unit tests
    testImplementation(libs.junit)
    testImplementation(libs.mockk)
    testImplementation(libs.mockwebserver)
    testImplementation(libs.kotlinx.coroutines.test)

    // Instrumented tests
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.espresso.intents)
    androidTestImplementation(libs.androidx.work.testing)
    androidTestImplementation(libs.mockwebserver)
    androidTestImplementation(libs.mockkandroid)
    androidTestImplementation(libs.kotlinx.coroutines.test)
    androidTestUtil(libs.androidx.test.orchestrator)

    // CameraX
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    implementation(libs.androidx.camera.video)
}

// Coverage is measured against the "full" flavor specifically: it's the only flavor that
// exercises every code path (all FEATURE_* flags true), and product flavors don't have a
// meaningful combined/aggregate coverage report the way a single-variant project would.
tasks.register<JacocoReport>("jacocoTestReport") {
    dependsOn("testFullDebugUnitTest")
    reports {
        xml.required.set(true)
        html.required.set(true)
    }

    val fileFilter = listOf(
        "**/R.class",
        "**/R$*.class",
        "**/BuildConfig.*",
        "**/Manifest*.*",
        "**/*Test*.*",
        "android/**/*.*",
        "androidx/**/*.*"
    )
    val debugTree = fileTree("${layout.buildDirectory.get()}/tmp/kotlin-classes/fullDebug") {
        exclude(fileFilter)
    }
    val mainSrc = "${project.projectDir}/src/main/java"

    sourceDirectories.setFrom(files(mainSrc))
    classDirectories.setFrom(files(debugTree))
    executionData.setFrom(fileTree(layout.buildDirectory.get()) {
        include("outputs/unit_test_code_coverage/fullDebugUnitTest/testFullDebugUnitTest.exec")
    })
}

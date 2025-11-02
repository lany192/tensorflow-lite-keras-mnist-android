pluginManagement {
    repositories {
        maven {
            url = uri("https://maven.aliyun.com/repository/public")
            content {
                includeGroupByRegex(".*")
            }
        }
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        maven {
            url = uri("https://maven.aliyun.com/repository/public")
            content {
                includeGroupByRegex(".*")
            }
        }
        google()
        mavenCentral()
    }
}

rootProject.name = "tensorflow-lite-keras-mnist-android"
include(":app")
 
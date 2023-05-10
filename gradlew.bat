@rem
@rem Copyright 2015 the original author or authors.
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem      https://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.
@rem

@if "%DEBUG%"=="" @echo off
@rem ##########################################################################
@rem
@rem  Gradle startup script for Windows
@rem
@rem ##########################################################################

@rem Set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" setlocal

@rem ##########################################################################
@rem Foundry-specific setup
@rem ##########################################################################

set _root_dir=%~dp0
set root_dir=%_root_dir:~0,-1%

if "%JEMMA%" == "" (
  setlocal EnableDelayedExpansion

  for /f %%i in ('git config remote.origin.url') DO (
    set git_remote=%%i
  )

  call set strip_before_host=!git_remote:*@=!

  call set _git_host_and_port_beg=%%strip_before_host:*/=%%
  call set _git_host_and_port_end=%%strip_before_host:!_git_host_and_port_beg!=%%
  call set git_host_and_port=!!_git_host_and_port_end:~0,-1!!

  call set _endpart=%%git_remote:*@=%%
  call set _firstpart=%%git_remote:!_endpart!=%%
  call set strip_after_userinfo=!!_firstpart:~0,-1!

  call set git_userinfo=%%strip_after_userinfo:*//=%%

  call set git_password=!!git_userinfo:*:=!
  call set _firstpart=%%git_userinfo:!git_password!=%%
  call set git_username=!!_firstpart:~0,-1!

  set GIT_REMOTE_HOST=!git_host_and_port!
  set GIT_REMOTE_USERNAME=!git_username!
  set GIT_REMOTE_PASSWORD=!git_password!

  @rem If not set, default to the URIs inferred from the git remote
  set ORG_GRADLE_PROJECT_transformsMavenProxyRepoUri=https://!git_host_and_port!/artifacts/api/legacy/mrp/authz/all
  set ORG_GRADLE_PROJECT_artifactsUri=https://!git_host_and_port!/artifacts/api
  set ORG_GRADLE_PROJECT_externalUri=https://!git_host_and_port!
  set ORG_GRADLE_PROJECT_transformsBearerToken=!git_password!

  set wrapperAuthGradleOptions=-Dgradle.wrapperUser=!GIT_REMOTE_USERNAME! -Dgradle.wrapperPassword=!GIT_REMOTE_PASSWORD!

  if ["%GRADLE_OPTS%"]==[""] set GRADLE_OPTS=
  set GRADLE_OPTS=%GRADLE_OPTS% !wrapperAuthGradleOptions!

  set transformsGradleDistributionUrl=!ORG_GRADLE_PROJECT_artifactsUri!/repositories/ri.gradle.distributions.artifacts.repository/contents/release/files/gradle-7.5.1-bin.zip
)

set input_filename=%root_dir%/gradle/wrapper/gradle-wrapper.template.properties
set output_filename=%root_dir%/gradle/wrapper/gradle-wrapper.properties
(for /f "tokens=1* delims=:" %%i in (%input_filename%) do (
  set s=%%i
  set s=!s:${transformsGradleDistributionUrl}=%transformsGradleDistributionUrl%!
  echo !s!
))>%output_filename%

@rem ##########################################################################

set DIRNAME=%~dp0
if "%DIRNAME%"=="" set DIRNAME=.
set APP_BASE_NAME=%~n0
set APP_HOME=%DIRNAME%

@rem Resolve any "." and ".." in APP_HOME to make it shorter.
for %%i in ("%APP_HOME%") do set APP_HOME=%%~fi

@rem Add default JVM options here. You can also use JAVA_OPTS and GRADLE_OPTS to pass JVM options to this script.
set DEFAULT_JVM_OPTS="-Xmx64m" "-Xms64m"

@rem Find java.exe
if defined JAVA_HOME goto findJavaFromJavaHome

set JAVA_EXE=java.exe
%JAVA_EXE% -version >NUL 2>&1
if %ERRORLEVEL% equ 0 goto execute

echo.
echo ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.
echo.
echo Please set the JAVA_HOME variable in your environment to match the
echo location of your Java installation.

goto fail

:findJavaFromJavaHome
set JAVA_HOME=%JAVA_HOME:"=%
set JAVA_EXE=%JAVA_HOME%/bin/java.exe

if exist "%JAVA_EXE%" goto execute

echo.
echo ERROR: JAVA_HOME is set to an invalid directory: %JAVA_HOME%
echo.
echo Please set the JAVA_HOME variable in your environment to match the
echo location of your Java installation.

goto fail

:execute
@rem Setup the command line

set CLASSPATH=%APP_HOME%\gradle\wrapper\gradle-wrapper.jar


@rem Execute Gradle
"%JAVA_EXE%" %DEFAULT_JVM_OPTS% %JAVA_OPTS% %GRADLE_OPTS% "-Dorg.gradle.appname=%APP_BASE_NAME%" -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*

:end
@rem End local scope for the variables with windows NT shell
if %ERRORLEVEL% equ 0 goto mainEnd

:fail
rem Set variable GRADLE_EXIT_CONSOLE if you need the _script_ return code instead of
rem the _cmd.exe /c_ return code!
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% equ 0 set EXIT_CODE=1
if not ""=="%GRADLE_EXIT_CONSOLE%" exit %EXIT_CODE%
exit /b %EXIT_CODE%

:mainEnd
if "%OS%"=="Windows_NT" endlocal

:omega

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$courseRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$lessonFiles = Get-ChildItem -LiteralPath $courseRoot -Filter "*.md" |
    Where-Object { $_.Name -match "^\d{2}_" } |
    Sort-Object Name

$errors = [System.Collections.Generic.List[string]]::new()

if ($lessonFiles.Count -ne 20) {
    $errors.Add("Expected 20 lesson files, found $($lessonFiles.Count).")
}

$expectedNumbers = 1..20
$actualNumbers = @($lessonFiles | ForEach-Object { [int]$_.Name.Substring(0, 2) })
if (Compare-Object $expectedNumbers $actualNumbers) {
    $errors.Add("Lesson numbers are not exactly 01 through 20.")
}

foreach ($lesson in $lessonFiles) {
    $content = Get-Content -Raw -LiteralPath $lesson.FullName
    if ($content -notmatch "(?m)^## 本课目标") {
        $errors.Add("$($lesson.Name): missing '本课目标'.")
    }
    if ($content -notmatch "实战|练习") {
        $errors.Add("$($lesson.Name): missing an exercise/practice section.")
    }
    if ($content -notmatch "本课验收|阶段 [A-E] 综合验收") {
        $errors.Add("$($lesson.Name): missing an acceptance section.")
    }
    $fenceCount = ([regex]::Matches($content, '(?m)^```')).Count
    if ($fenceCount % 2 -ne 0) {
        $errors.Add("$($lesson.Name): unbalanced Markdown code fences.")
    }
}

$markdownFiles = Get-ChildItem -LiteralPath $courseRoot -Filter "*.md"
foreach ($source in $markdownFiles) {
    $content = Get-Content -Raw -LiteralPath $source.FullName
    foreach ($match in [regex]::Matches($content, "\[[^\]]+\]\(([^)]+)\)")) {
        $target = $match.Groups[1].Value
        if ($target -match "^(https?://|#)") {
            continue
        }
        $resolved = Join-Path $source.DirectoryName $target
        if (-not (Test-Path -LiteralPath $resolved)) {
            $errors.Add("$($source.Name): broken local link '$target'.")
        }
    }
}

$readmePath = Join-Path $courseRoot "README.md"
$readme = Get-Content -Raw -LiteralPath $readmePath
$readmeLessonLinks = ([regex]::Matches($readme, "(?m)^- \[\d{2}：")).Count
if ($readmeLessonLinks -ne 20) {
    $errors.Add("README navigation contains $readmeLessonLinks lesson links instead of 20.")
}

$syllabusPath = Join-Path $courseRoot "完整教学大纲.md"
$syllabus = Get-Content -Raw -LiteralPath $syllabusPath
$syllabusChapters = ([regex]::Matches($syllabus, "(?m)^### \d{2} ")).Count
if ($syllabusChapters -ne 20) {
    $errors.Add("Syllabus contains $syllabusChapters chapters instead of 20.")
}

$summary = [pscustomobject]@{
    Lessons = $lessonFiles.Count
    MarkdownFiles = $markdownFiles.Count
    ReadmeLessonLinks = $readmeLessonLinks
    SyllabusChapters = $syllabusChapters
    TotalLessonBytes = ($lessonFiles | Measure-Object Length -Sum).Sum
    Errors = $errors.Count
}
$summary | Format-List

if ($errors.Count -gt 0) {
    Write-Host "Validation errors:" -ForegroundColor Red
    $errors | ForEach-Object { Write-Host "- $_" -ForegroundColor Red }
    exit 1
}

Write-Host "Course validation passed." -ForegroundColor Green

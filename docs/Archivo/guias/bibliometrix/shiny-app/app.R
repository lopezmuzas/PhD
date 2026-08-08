library(shiny)
library(shinydashboard)
library(bibliometrix)
library(DT)
library(ggplot2)
library(plotly)
library(dplyr)

# ── UI ────────────────────────────────────────────────────────────────────────
ui <- dashboardPage(
  skin = "blue",

  dashboardHeader(title = "Bibliometrix Explorer"),

  dashboardSidebar(
    sidebarMenu(
      menuItem("Inicio",          tabName = "home",    icon = icon("home")),
      menuItem("Cargar datos",    tabName = "upload",  icon = icon("upload")),
      menuItem("Resumen",         tabName = "summary", icon = icon("info-circle")),
      menuItem("Autores",         tabName = "authors", icon = icon("users")),
      menuItem("Fuentes",         tabName = "sources", icon = icon("book")),
      menuItem("Palabras clave",  tabName = "keywords",icon = icon("tags")),
      menuItem("Co-citación",     tabName = "cocit",   icon = icon("project-diagram")),
      menuItem("Datos raw",       tabName = "rawdata", icon = icon("table"))
    )
  ),

  dashboardBody(
    tabItems(

      # ── Inicio ──────────────────────────────────────────────────────────────
      tabItem(tabName = "home",
        fluidRow(
          box(width = 12, status = "primary", solidHeader = TRUE,
            title = "Bienvenido a Bibliometrix Explorer",
            h4("Herramienta de análisis bibliométrico basada en el paquete ", code("bibliometrix")),
            p("Pasos para empezar:"),
            tags$ol(
              tags$li("Ve a ", strong("Cargar datos"), " y sube tu fichero exportado desde Web of Science, Scopus, PubMed, etc."),
              tags$li("Explora el ", strong("Resumen"), " general de la colección."),
              tags$li("Analiza ", strong("Autores"), ", ", strong("Fuentes"), " y ", strong("Palabras clave"), ".")
            ),
            hr(),
            p("Formatos soportados: BibTeX (.bib), RIS (.ris), Web of Science (.txt), Scopus CSV (.csv), PubMed (.txt)")
          )
        )
      ),

      # ── Cargar datos ────────────────────────────────────────────────────────
      tabItem(tabName = "upload",
        fluidRow(
          box(width = 6, status = "warning", solidHeader = TRUE,
            title = "Subir archivo bibliográfico",
            fileInput("file", "Selecciona el archivo",
                      accept = c(".bib", ".ris", ".txt", ".csv")),
            selectInput("dbsource", "Base de datos origen",
                        choices = c("wos", "scopus", "pubmed", "cochrane",
                                    "openalex", "lens", "csci", "isi")),
            selectInput("format", "Formato del archivo",
                        choices = c("bibtex", "plaintext", "csv", "ris")),
            actionButton("load_btn", "Cargar", icon = icon("play"),
                         class = "btn-primary btn-lg")
          ),
          box(width = 6, status = "info", solidHeader = TRUE,
            title = "Estado de carga",
            verbatimTextOutput("load_status")
          )
        )
      ),

      # ── Resumen ─────────────────────────────────────────────────────────────
      tabItem(tabName = "summary",
        fluidRow(
          valueBoxOutput("vbox_docs",    width = 3),
          valueBoxOutput("vbox_authors", width = 3),
          valueBoxOutput("vbox_sources", width = 3),
          valueBoxOutput("vbox_years",   width = 3)
        ),
        fluidRow(
          box(width = 12, title = "Producción científica anual",
              status = "primary", solidHeader = TRUE,
              plotlyOutput("annual_plot", height = "350px"))
        ),
        fluidRow(
          box(width = 12, title = "Resumen detallado",
              status = "info", solidHeader = TRUE,
              verbatimTextOutput("summary_text"))
        )
      ),

      # ── Autores ─────────────────────────────────────────────────────────────
      tabItem(tabName = "authors",
        fluidRow(
          box(width = 12, title = "Top autores más productivos",
              status = "primary", solidHeader = TRUE,
              sliderInput("top_authors", "Número de autores a mostrar:",
                          min = 5, max = 50, value = 20, step = 5),
              plotlyOutput("authors_plot", height = "400px"))
        ),
        fluidRow(
          box(width = 12, title = "Tabla de autores",
              status = "info", solidHeader = TRUE,
              DTOutput("authors_table"))
        )
      ),

      # ── Fuentes ─────────────────────────────────────────────────────────────
      tabItem(tabName = "sources",
        fluidRow(
          box(width = 12, title = "Top fuentes / revistas",
              status = "primary", solidHeader = TRUE,
              sliderInput("top_sources", "Número de fuentes a mostrar:",
                          min = 5, max = 30, value = 15, step = 5),
              plotlyOutput("sources_plot", height = "400px"))
        )
      ),

      # ── Palabras clave ──────────────────────────────────────────────────────
      tabItem(tabName = "keywords",
        fluidRow(
          box(width = 12, title = "Palabras clave más frecuentes",
              status = "primary", solidHeader = TRUE,
              sliderInput("top_kw", "Número de palabras clave:",
                          min = 10, max = 100, value = 30, step = 10),
              radioButtons("kw_type", "Tipo de palabras clave:",
                           choices = c("Autor (DE)" = "DE", "Plus (ID)" = "ID"),
                           inline = TRUE),
              plotlyOutput("keywords_plot", height = "450px"))
        )
      ),

      # ── Co-citación ─────────────────────────────────────────────────────────
      tabItem(tabName = "cocit",
        fluidRow(
          box(width = 12, status = "warning", solidHeader = TRUE,
            title = "Análisis de co-citación (referencias)",
            p("Muestra las referencias más co-citadas en la colección."),
            sliderInput("top_cocit", "Top referencias:", min = 5, max = 30, value = 15),
            DTOutput("cocit_table"))
        )
      ),

      # ── Datos raw ───────────────────────────────────────────────────────────
      tabItem(tabName = "rawdata",
        fluidRow(
          box(width = 12, title = "Datos bibliográficos cargados",
              status = "primary", solidHeader = TRUE,
              DTOutput("raw_table"))
        )
      )
    )
  )
)

# ── SERVER ────────────────────────────────────────────────────────────────────
server <- function(input, output, session) {

  # Datos reactivos
  bib_data <- reactiveVal(NULL)
  results  <- reactiveVal(NULL)

  # Cargar archivo
  observeEvent(input$load_btn, {
    req(input$file)
    tryCatch({
      M <- convert2df(
        file     = input$file$datapath,
        dbsource = input$dbsource,
        format   = input$format
      )
      bib_data(M)
      res <- biblioAnalysis(M, sep = ";")
      results(res)
      output$load_status <- renderText({
        paste0("✅ Cargados ", nrow(M), " documentos correctamente.\n",
               "Columnas disponibles: ", paste(names(M), collapse = ", "))
      })
    }, error = function(e) {
      output$load_status <- renderText(paste("❌ Error:", e$message))
    })
  })

  # ── Value boxes ────────────────────────────────────────────────────────────
  output$vbox_docs <- renderValueBox({
    req(results())
    valueBox(results()$Articles, "Documentos", icon = icon("file-alt"), color = "blue")
  })
  output$vbox_authors <- renderValueBox({
    req(results())
    valueBox(results()$nAuthors, "Autores", icon = icon("users"), color = "green")
  })
  output$vbox_sources <- renderValueBox({
    req(results())
    valueBox(results()$Journals, "Fuentes", icon = icon("book"), color = "purple")
  })
  output$vbox_years <- renderValueBox({
    req(bib_data())
    yrs <- range(bib_data()$PY, na.rm = TRUE)
    valueBox(paste(yrs[1], "–", yrs[2]), "Período", icon = icon("calendar"), color = "orange")
  })

  # ── Producción anual ───────────────────────────────────────────────────────
  output$annual_plot <- renderPlotly({
    req(bib_data())
    df <- bib_data() %>%
      filter(!is.na(PY)) %>%
      count(PY) %>%
      rename(Year = PY, Articles = n)
    p <- ggplot(df, aes(x = Year, y = Articles)) +
      geom_col(fill = "#3c8dbc") +
      geom_line(color = "#e74c3c", linewidth = 1) +
      labs(x = "Año", y = "Artículos", title = "Producción anual") +
      theme_minimal()
    ggplotly(p)
  })

  # ── Resumen texto ──────────────────────────────────────────────────────────
  output$summary_text <- renderPrint({
    req(results())
    summary(results(), k = 10, pause = FALSE)
  })

  # ── Autores ────────────────────────────────────────────────────────────────
  output$authors_plot <- renderPlotly({
    req(results())
    top <- input$top_authors
    au  <- sort(results()$Authors, decreasing = TRUE)[1:min(top, length(results()$Authors))]
    df  <- data.frame(Author = names(au), Articles = as.integer(au))
    p <- ggplot(df, aes(x = reorder(Author, Articles), y = Articles)) +
      geom_col(fill = "#27ae60") + coord_flip() +
      labs(x = NULL, y = "Artículos") + theme_minimal()
    ggplotly(p)
  })

  output$authors_table <- renderDT({
    req(results())
    au <- sort(results()$Authors, decreasing = TRUE)
    datatable(data.frame(Autor = names(au), Artículos = as.integer(au)),
              options = list(pageLength = 15))
  })

  # ── Fuentes ────────────────────────────────────────────────────────────────
  output$sources_plot <- renderPlotly({
    req(results())
    top <- input$top_sources
    so  <- sort(results()$Sources, decreasing = TRUE)[1:min(top, length(results()$Sources))]
    df  <- data.frame(Source = names(so), Articles = as.integer(so))
    p <- ggplot(df, aes(x = reorder(Source, Articles), y = Articles)) +
      geom_col(fill = "#8e44ad") + coord_flip() +
      labs(x = NULL, y = "Artículos") + theme_minimal()
    ggplotly(p)
  })

  # ── Palabras clave ─────────────────────────────────────────────────────────
  output$keywords_plot <- renderPlotly({
    req(bib_data())
    col <- input$kw_type
    if (!col %in% names(bib_data())) {
      return(plotly_empty() %>% layout(title = paste("Columna", col, "no disponible")))
    }
    kw <- tableTag(bib_data(), col, sep = ";")
    top <- min(input$top_kw, length(kw))
    df  <- data.frame(Keyword = names(kw)[1:top], Freq = as.integer(kw[1:top]))
    p <- ggplot(df, aes(x = reorder(Keyword, Freq), y = Freq)) +
      geom_col(fill = "#e67e22") + coord_flip() +
      labs(x = NULL, y = "Frecuencia") + theme_minimal()
    ggplotly(p)
  })

  # ── Co-citación ────────────────────────────────────────────────────────────
  output$cocit_table <- renderDT({
    req(bib_data())
    tryCatch({
      cr <- citations(bib_data(), field = "article", sep = ";")
      top <- input$top_cocit
      df  <- as.data.frame(cr$Cited[1:min(top, nrow(cr$Cited)), ])
      datatable(df, options = list(pageLength = 10))
    }, error = function(e) {
      datatable(data.frame(Info = paste("No disponible:", e$message)))
    })
  })

  # ── Datos raw ──────────────────────────────────────────────────────────────
  output$raw_table <- renderDT({
    req(bib_data())
    cols <- intersect(c("AU","TI","SO","PY","DE","AB","TC"), names(bib_data()))
    datatable(bib_data()[, cols, drop = FALSE],
              options = list(pageLength = 10, scrollX = TRUE))
  })
}

shinyApp(ui, server)

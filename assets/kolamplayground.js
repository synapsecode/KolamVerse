let generator;

function setup() {
    const canvas = createCanvas(windowWidth * 0.6, windowHeight);
    canvas.parent("canvas-holder");
    generator = new Pattern();
    makePattern();

    document.getElementById("refresh").onclick = () => {
        generator.tileSize = +document.getElementById("tileSize").value;
        generator.gridSize = +document.getElementById("gridSize").value;
        generator.angle = parseFloat(document.getElementById("rotation").value);
        generator.fillFactor = parseFloat(document.getElementById("density").value);
        generator.symMode = parseInt(document.getElementById("symmetryMode").value);
        makePattern();
    };

    noLoop();
}

function draw() {
    background(generator.bg);
    generator.show();
}

function windowResized() {
    resizeCanvas(windowWidth * 0.6, windowHeight);
    makePattern();
}

function makePattern() {
    generator.initTiles();
    generator.populate();
    generator.renderTile();
    redraw();
}

class Pattern {
    constructor() {
        this.tileSize = 45;
        this.gridSize = 10;
        this.angle = QUARTER_PI;
        this.fillFactor = 0.6;
        this.symMode = 4;
        this.bg = color(0, 0, 0);

        this.data = [];
        this.symData = [];
        this.pg = null;
    }

    initTiles() {
        this.pg = createGraphics(this.tileSize * this.gridSize, this.tileSize * this.gridSize);
        this.data = [];
        this.symData = [];
        for (let i = 0; i <= this.gridSize; i++) {
            let row = new Array(this.gridSize + 1).fill(1);
            this.data.push([...row]);
            this.symData.push([...row]);
        }
    }

    populate() {
        for (let i = 0; i < this.symData.length; i++) {
            for (let j = 0; j < this.symData.length / 2; j++) {
                let val = (random(1) > this.fillFactor) ? 1 : 0;
                this.applySymmetry(i, j, val);
            }
        }
    }

    applySymmetry(i, j, v) {
        const n = this.symData.length - 1;
        const cx = n / 2;
        const cy = n / 2;

        let dx = i - cx;
        let dy = j - cy;

        for (let k = 0; k < this.symMode; k++) {
            let angle = TWO_PI * k / this.symMode;
            let x = Math.round(cx + dx * cos(angle) - dy * sin(angle));
            let y = Math.round(cy + dx * sin(angle) + dy * cos(angle));
            if (x >= 0 && x <= n && y >= 0 && y <= n) {
                this.symData[x][y] = v;
            }
        }
    }

    renderTile() {
        this.pg.background(this.bg);
        this.pg.noFill();
        this.pg.stroke(255);
        this.pg.strokeWeight(5);
        this.pg.strokeCap(SQUARE);

        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                if ((i + j) % 2 === 0) {
                    let tl = this.tileSize / 2 * this.symData[i][j];
                    let tr = this.tileSize / 2 * this.symData[i + 1][j];
                    let br = this.tileSize / 2 * this.symData[i + 1][j + 1];
                    let bl = this.tileSize / 2 * this.symData[i][j + 1];

                    this.pg.rect(
                        i * this.tileSize,
                        j * this.tileSize,
                        this.tileSize,
                        this.tileSize,
                        tl, tr, br, bl
                    );

                    this.pg.point(
                        i * this.tileSize + this.tileSize / 2,
                        j * this.tileSize + this.tileSize / 2
                    );
                }
            }
        }
    }

    show() {
        push();
        translate(width / 2, height / 2);
        rotate(this.angle);
        imageMode(CENTER);
        image(this.pg, 0, 0);
        pop();
    }
}